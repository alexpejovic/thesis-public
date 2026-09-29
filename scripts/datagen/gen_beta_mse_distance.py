"""How far each beta-encoder's loss landscape is from the MSE landscape.

One run = one 3-parameter subset of the eight neuronal parameters, on one ideal
trace seed. It sweeps a dense grid over those three parameters, and on the SAME
simulated traces evaluates

  * ``mse``  -- mean squared error between ZScoreGlobal-normalized traces, and
  * ``feat`` -- the latent-L2 encoder loss, once per encoder in the manifest.

Both landscapes are then min-max'd to [0, 1] and reduced to a single number per
encoder::

    distance = mean((unit(feat) - unit(mse)) ** 2)

which is the quantitative version of the two-surface overlay that
``scripts/plotting/compare_encoder_to_mse2d.py`` already draws.

The encoders form a ladder in ``beta``, replicated over several VAE *training*
seeds. One (beta, training seed) draw is noisy -- a single 3-D landscape gave a
clean beta=0 separation but no monotone trend above it -- so every beta carries
several independently trained encoders and the analysis averages over them. The
encoder set is pinned in a manifest (see ``--build_manifest``) rather than
discovered per task, so all 56 array tasks score the identical set even if more
models finish training while the array is draining.

This is ``generate_losses.py`` stripped to exactly that: no ``b1_loss``/``b2_loss``
(identically zero under ZScoreGlobal anyway), no allen features, no ``feat_loss``
ladder -- only the ``feat_loss: mean`` case -- and one encoder replaced by five.

Every default comes from ``configs/beta_mse.yaml``; the parameter sets and the
canonical parameter order come from ``param_groups``/``sim``, so a set is never
defined twice. Fan-out is the ``beta_mse_distance`` study, not a bash loop:

    cd encoding/slurm_scripts
    ./submit_study.sh beta_mse_distance --dry-run

Usage (from encoding/):
    python scripts/datagen/gen_beta_mse_distance.py --build_manifest
    python scripts/datagen/gen_beta_mse_distance.py --set_key gLeak+eLeak+gNa
    python scripts/datagen/gen_beta_mse_distance.py \
        --params Na_gNa K_gK Km_gKm --seed 0 --opt_samples 91 --opt_bounds -1.5 1.5
"""

import argparse
import json
import os
import sys
import time
from itertools import batched, product
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import yaml
from jaxley import Compartment
from scipy.stats import spearmanr

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

import param_groups as pg  # noqa: E402
from helper import (  # noqa: E402
    compress_traces,
    get_beta_mse_config,
    get_datagen_config,
    get_param_bounds,
    get_trace_normal_and_denormal,
    insert_channels,
    set_const_params,
    unit,
    unit_robust,
)
from logger import get_logger, log_commit  # noqa: E402
from sim import ALL_PARAMS, canonical, check_param_order  # noqa: E402
from train_comp import (  # noqa: E402
    _backward_transform_params,
    _forward_transform_params,
    _get_transform_list,
    get_traces_to_match_rand,
    set_frozen_params,
)
from vae.conv_vae import TimeVAEBase  # noqa: E402
from vae.vae_helper import get_vae_config, load_encoder  # noqa: E402

jax.config.update("jax_enable_x64", True)


def assert_backend_matches_allocation() -> str:
    """Fail loudly when SLURM gave us a GPU but JAX quietly fell back to CPU.

    jaxlib without the `cuda13` extra prints a warning to stderr and runs on CPU
    anyway -- roughly 20x slower here. That cost 6 A100s x 17 h of useless work
    before anyone noticed, because a slow job and a correct job look identical from
    the outside. There is no flag to forget: if the scheduler allocated a device, a
    CPU backend is by definition wrong.
    """
    backend = jax.default_backend()
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if allocated and backend == "cpu":
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES={allocated!r} but jax.default_backend()=='cpu'.\n"
            "This venv is missing the CUDA plugin. Fix it with\n"
            "  make requirements-extras     (installs the cuda13 extra)\n"
            "or point PY at a venv that already has it."
        )
    return backend

# The ladder (`ladder_betas`, `ladder_template`) lives in configs/beta_mse.yaml,
# so the definition of "comparable" is stated once and a stray run with a
# different latent_dim can never silently join the study.


def discover_encoders(
    channel_type: str, ladder_betas: list[float], ladder_template: dict
) -> list[dict]:
    """Every trained run on the ladder, as {beta, train_seed, datestr} records.

    Sorted by (beta, train_seed) so the ordering is deterministic and the parallel
    `distances` array in the output is interpretable without a join.
    """
    base = Path(f"./data/models/vae/{channel_type}/{ladder_template['vae_type']}")
    found = []
    for run in sorted(base.iterdir()) if base.is_dir() else []:
        if not run.is_dir() or run.is_symlink():
            continue
        cfg_path = run / "config.yaml"
        # A run still training has no final-model.eqx; skip it rather than crash
        # halfway through a 14-hour sweep.
        if not cfg_path.exists() or not (run / "final-model.eqx").exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        beta = cfg.get("beta", cfg.get("beta_norm"))
        if beta is None or not any(abs(beta - b) < 1e-12 for b in ladder_betas):
            continue
        if any(cfg.get(k) != v for k, v in ladder_template.items()):
            continue
        found.append(
            {"beta": float(beta), "train_seed": int(cfg["seed"]), "datestr": run.name}
        )

    # One model per (beta, training seed). The template is a config match, and
    # config is not identity: runs from before 2026-08-17 share these
    # hyperparameters but predate the current ZScoreGlobal normalizer and the
    # masking option, so they are different models under the same settings --
    # e.g. the other study's `encoder-zscore` (2026_08_12_15_20_26_225838) matches
    # this template exactly. Keeping the newest datestr per cell picks the run
    # built by current code; without this the ladder silently gains a duplicate
    # rung trained by a different commit.
    #
    # datestr is `%Y_%m_%d_%H_%M_%S_%f`, so lexicographic max IS chronological max.
    best: dict[tuple[float, int], dict] = {}
    for r in found:
        cell = (r["beta"], r["train_seed"])
        prev = best.get(cell)
        if prev is None or r["datestr"] > prev["datestr"]:
            best[cell] = r
    dropped = [r for r in found if best[(r["beta"], r["train_seed"])] is not r]
    for r in dropped:
        print(
            f"  superseded: beta={r['beta']:g} seed={r['train_seed']} "
            f"{r['datestr']}"
        )
    return sorted(best.values(), key=lambda r: (r["beta"], r["train_seed"]))


def load_manifest(path: Path, ladder_template: dict) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Build it once with:\n"
            "  python scripts/datagen/gen_beta_mse_distance.py --build_manifest"
        )
    records = json.loads(path.read_text())["encoders"]
    missing = [
        r["datestr"]
        for r in records
        if not Path(
            f"./data/models/vae/{ladder_template['trace_data']}/"
            f"{ladder_template['vae_type']}/{r['datestr']}/final-model.eqx"
        ).exists()
    ]
    if missing:
        raise SystemExit(f"manifest references missing checkpoints: {missing}")
    return records

def all_params(config: dict, channel_type: str) -> list[str]:
    """The canonical parameter order -- the config's, which is what the simulator
    iterates in (`training_bounds.keys()`).

    A flat grid axis `i` is ALL_PARAMS-ordered, not CLI-ordered, so a reordered
    datagen.yaml must be a loud failure rather than a transposed landscape.
    `sim.check_param_order` is that assertion; it is only meaningful for the
    channel type ALL_PARAMS describes.
    """
    got = list(get_param_bounds(config["param_bounds"], channel_type))
    if channel_type == "Pospischil":
        check_param_order(got)
    return got


def _sweep(
    channel_type: str,
    params: list[str],
    encoder_records: list[dict],
    seed: int,
    opt_bounds: tuple[float, float],
    opt_samples: int,
    batch_size: int,
    logger,
):
    """Simulate the grid once; return (mse, feats, axis, ideal_v_range)."""
    config = get_datagen_config()
    dt = config["dt"]
    t_max = config["t_max"]
    t_filter = config["t_filter"]
    avg_scale = config["avg_scale"]
    current_params = config["current"]

    comp = Compartment()
    comp = insert_channels(comp, channel_type)
    comp = set_const_params(config["param_consts"], channel_type, comp)
    comp.record("v")

    param_bounds = get_param_bounds(config["param_bounds"], channel_type)
    freeze_params = [p for p in param_bounds if p not in params]
    training_bounds = {k: v for k, v in param_bounds.items() if k not in freeze_params}
    freeze_bounds = {k: v for k, v in param_bounds.items() if k in freeze_params}
    if list(training_bounds) != params:
        raise RuntimeError(f"swept axes {list(training_bounds)} != requested {params}")
    transform_list = _get_transform_list(training_bounds)

    # PRNG stream copied from generate_losses.py, including the first split that is
    # spent on `amps`, so the ideal params drawn for a given seed match the ones
    # every landscape already on disk was built from.
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    amps = get_traces_to_match_rand(channel_type, config, subkey)[1]

    ideal_params_dict = {}
    for param_name, bounds in param_bounds.items():
        key, subkey = jax.random.split(key)
        bound_center = (bounds[0] + bounds[1]) / 2
        q1 = (bounds[0] + bound_center) / 2
        q2 = (bound_center + bounds[1]) / 2
        ideal_params_dict[param_name] = jax.random.uniform(
            subkey, minval=q1, maxval=q2
        )

    comp = set_frozen_params(freeze_bounds, comp, ideal_params_dict)
    comp.init_states(delta_t=dt)

    transformed_ideals = {
        k: float(v)
        for k, v in zip(
            training_bounds.keys(),
            _backward_transform_params(
                [v for k, v in ideal_params_dict.items() if k not in freeze_params],
                transform_list,
            ),
        )
    }

    def simulate_network(p, amp):
        p = _forward_transform_params(p, transform_list)
        current = jx.step_current(
            i_delay=current_params["delay"],
            i_dur=current_params["duration"],
            i_amp=amp,
            delta_t=dt,
            t_max=t_max,
        )
        param_state = None
        for i, param in enumerate(training_bounds.keys()):
            param_state = comp.data_set(param, p[i], param_state)
        data_stimuli = comp.data_stimulate(current, None)
        return jnp.array(
            jx.integrate(
                comp,
                data_stimuli=data_stimuli,
                param_state=param_state,
                delta_t=dt,
                t_max=t_max,
            ).flatten()
        )

    simulate_amps = jax.vmap(simulate_network, in_axes=(None, 0))

    ideal_voltages = simulate_amps(list(transformed_ideals.values()), amps)
    ideal_voltages = compress_traces(ideal_voltages, t_filter, dt, avg_scale)
    ideal_v_range = float(jnp.max(ideal_voltages) - jnp.min(ideal_voltages))
    logger.info(f"ideal trace voltage range: {ideal_v_range:.2f} mV")

    # The normalizer belongs to the checkpoints, not to this script. All five must
    # agree, since a distance between differently-normalized landscapes is
    # meaningless.
    normalizer_types = {
        get_vae_config(channel_type, TimeVAEBase, r["datestr"])["normalizer_type"]
        for r in encoder_records
    }
    if len(normalizer_types) != 1:
        raise RuntimeError(f"encoders disagree on normalizer: {normalizer_types}")
    normalizer_type = normalizer_types.pop()
    if normalizer_type != "ZScoreGlobal":
        raise RuntimeError(
            f"expected ZScoreGlobal, got {normalizer_type!r}; the MSE reference and "
            "the encoder inputs must share one normalizer"
        )
    logger.info(f"normalizer: {normalizer_type}")

    normalizer, _ = get_trace_normal_and_denormal(normalizer_type, channel_type, config)
    normalize = jax.vmap(normalizer)

    encoders = [
        jax.vmap(
            load_encoder(
                TimeVAEBase,
                channel_type,
                ideal_voltages.shape[1],
                r["datestr"],
                epoch=None,
            )
        )
        for r in encoder_records
    ]
    logger.info(
        f"{len(encoders)} encoders: "
        + ", ".join(f"b{r['beta']:g}/s{r['train_seed']}" for r in encoder_records)
    )

    # Normalize in f64 and cast only for the encoder, whose weights are f32.
    normalized_ideal = normalize(ideal_voltages)
    ideal_means = [enc(normalized_ideal.astype(jnp.float32))[0] for enc in encoders]

    @jax.jit
    @jax.vmap
    def loss_fn(p):
        v = simulate_amps(p, amps)
        v = compress_traces(v, t_filter, dt, avg_scale)
        v = jnp.nan_to_num(v)
        nv = normalize(v)
        mse = jnp.mean((normalized_ideal - nv) ** 2)
        nv32 = nv.astype(jnp.float32)
        feats = jnp.stack(
            [
                jnp.mean((im - enc(nv32)[0]) ** 2).astype(jnp.float64)
                for enc, im in zip(encoders, ideal_means)
            ]
        )
        return mse, feats

    axis = np.linspace(float(opt_bounds[0]), float(opt_bounds[1]), int(opt_samples))
    total = int(opt_samples) ** len(params)
    # product() varies the LAST axis fastest, i.e. C order, so axis i of the
    # reshaped landscape is params[i].
    mses, feats = [], []
    done = 0
    t0 = time.time()
    for chunk in batched(product(*[axis] * len(params)), batch_size):
        m, f = loss_fn(jnp.asarray(chunk))
        mses.append(np.asarray(m))
        feats.append(np.asarray(f))
        done += len(chunk)
        if done % (batch_size * 50) < batch_size or done == total:
            logger.info(f"{done}/{total} grid points ({time.time() - t0:.0f}s)")

    mse = np.concatenate(mses)
    feat = np.concatenate(feats, axis=0).T  # (n_encoders, total)
    if mse.shape != (total,) or feat.shape != (len(encoder_records), total):
        raise RuntimeError(f"grid shape {mse.shape}/{feat.shape} != {total}")
    return mse, feat, axis, ideal_v_range, transformed_ideals


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def main(argv=None) -> int:
    # Hand-written argparse, not the `_parse_args`-from-YAML auto-flag pattern:
    # --opt_bounds is two-valued and --params is variadic, and `type=type(v)` on a
    # list default becomes `type=list`, which shreds "-1.5 1.5" into single
    # characters. Same deliberate deviation as gen_region_success_scores.py. Every
    # default still comes from the YAML rather than a literal.
    cfg = get_beta_mse_config()

    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--set_key",
        help="parameter set as a param_groups key, e.g. gLeak+eLeak+gNa; this is "
        "what the study axis iterates",
    )
    src.add_argument(
        "--params",
        nargs="+",
        help="parameter names, instead of --set_key; order is irrelevant, they "
        "are canonicalised",
    )
    src.add_argument(
        "--list_sets",
        action="store_true",
        help="print every parameter set key and its parameters, then exit",
    )
    src.add_argument(
        "--build_manifest",
        action="store_true",
        help="scan for ladder encoders, pin them to the manifest, and exit. Run "
        "this ONCE after training finishes; every array task then scores the "
        "identical encoder set.",
    )
    src.add_argument(
        "--list_encoders",
        action="store_true",
        help="print the discovered ladder encoders and exit",
    )
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--channel_type", default=cfg["channel_type"])
    ap.add_argument("--opt_bounds", nargs=2, type=float, default=cfg["opt_bounds"])
    ap.add_argument("--opt_samples", type=int, default=cfg["opt_samples"])
    ap.add_argument("--batch_size", type=int, default=cfg["batch_size"])
    ap.add_argument(
        "--save_losses",
        action="store_true",
        help="also store the raw loss grids, for the scatter plots",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="recompute even if this unit's record already exists",
    )
    ap.add_argument(
        "--allow_unbalanced",
        action="store_true",
        help="--build_manifest only: pin a ladder whose betas carry different "
        "numbers of training seeds",
    )
    ap.add_argument(
        "--encoder_manifest",
        type=Path,
        default=Path(cfg["encoder_manifest"]),
        help=f"Default: {cfg['encoder_manifest']}",
    )
    args = ap.parse_args(argv)

    ladder_betas = [float(b) for b in cfg["ladder_betas"]]
    ladder_template = cfg["ladder_template"]
    out_base = Path(cfg["save_path"])

    if args.list_sets:
        for key in pg.STUDY_SETS:
            print(f"{key}\t{' '.join(pg.params_for(key))}")
        return 0

    backend = assert_backend_matches_allocation()

    config = get_datagen_config()
    all_known = all_params(config, args.channel_type)

    if args.list_encoders or args.build_manifest:
        found = discover_encoders(args.channel_type, ladder_betas, ladder_template)
        by_beta = {}
        for r in found:
            by_beta.setdefault(r["beta"], []).append(r["train_seed"])
        for b in ladder_betas:
            seeds = by_beta.get(b, [])
            print(f"beta={b:<5g} n={len(seeds)}  seeds={seeds}")
        print(f"total {len(found)} encoders")
        if args.list_encoders:
            for r in found:
                print(
                    f"  beta={r['beta']:<5g} seed={r['train_seed']:<6d} "
                    f"{r['datestr']}"
                )
            return 0

        if not found:
            raise SystemExit("no ladder encoders found")

        # An unbalanced ladder would make the per-beta means rest on different
        # numbers of draws, which is exactly the variance this study exists to
        # remove. Refuse rather than silently average 1 against 4; --pilot and
        # every array task read whatever this pins.
        counts = {len(by_beta.get(b, [])) for b in ladder_betas}
        if len(counts) != 1:
            msg = (
                f"unbalanced ladder, per-beta counts {sorted(counts)}; some betas "
                "would be averaged over fewer training seeds"
            )
            if not args.allow_unbalanced:
                raise SystemExit(
                    f"REFUSING: {msg}.\nPass --allow_unbalanced to pin it anyway."
                )
            print(f"WARNING: {msg} (--allow_unbalanced)")

        args.encoder_manifest.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            args.encoder_manifest,
            {
                "encoders": found,
                "ladder_betas": ladder_betas,
                "template": ladder_template,
                "channel_type": args.channel_type,
            },
        )
        print(f"wrote {args.encoder_manifest}")
        return 0

    # `param_groups` owns the 56 sets and `sim.canonical` owns the ordering, so a
    # set is never defined twice. --set_key resolves to the same full parameter
    # names in the same canonical order that --params produces, which is what
    # keeps every record already on disk addressable.
    if args.set_key is not None:
        try:
            params = canonical(pg.params_for(args.set_key))
        except KeyError as e:
            raise SystemExit(f"{e}; see --list_sets")
    else:
        unknown = [q for q in args.params if q not in all_known]
        if unknown:
            raise SystemExit(f"unknown parameter(s) {unknown}; known: {all_known}")
        try:
            params = canonical(args.params)
        except ValueError as e:
            raise SystemExit(str(e))
    try:
        set_key = pg.set_key(params)
    except KeyError:
        # param_groups' short names cover the Pospischil set only; the key is a
        # convenience for joining records, not a precondition for sweeping.
        set_key = "+".join(params)

    out_dir = out_base / args.channel_type
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{'+'.join(params)}_seed{args.seed}_n{args.opt_samples}"
    record_path = out_dir / f"{stem}.json"

    # Cheap resume, complementary to the study ledger: an array task that comes
    # back over a finished unit exits 0 without paying for the grid again. The
    # ledger in data/study-runs.jsonl is still what decides what gets submitted.
    if record_path.exists() and not args.force:
        print(f"skip {set_key} seed={args.seed} n={args.opt_samples}  (already done)")
        return 0

    encoder_records = load_manifest(args.encoder_manifest, ladder_template)

    logger = get_logger(str(out_dir / f"{stem}.log"), name="gen_beta_mse_distance")
    log_commit(logger, "../")
    logger.info(
        f"params={params} seed={args.seed} bounds={tuple(args.opt_bounds)} "
        f"samples={args.opt_samples} backend={backend}"
    )

    t0 = time.time()
    mse, feat, axis, ideal_v_range, ideals = _sweep(
        args.channel_type,
        params,
        encoder_records,
        args.seed,
        args.opt_bounds,
        args.opt_samples,
        args.batch_size,
        logger,
    )

    mse_u = unit(mse)
    # Parallel to encoder_records, so (beta, train_seed, datestr, distance) needs
    # no join. Averaging over train_seed within a beta is the plotting script's job.
    feat_u = [unit(feat[i]) for i in range(len(encoder_records))]
    distances = [float(np.mean((f - mse_u) ** 2)) for f in feat_u]

    # Min-max normalization is outlier-sensitive: one diverging grid point squashes
    # the rest of a landscape toward 0 and shrinks its distance to anything.
    # `p98_span` is how much of [0, 1] the middle 98% of a normalized landscape
    # actually occupies -- near 1.0 is healthy, near 0 means a single extreme point
    # is setting the scale and that landscape's distance should not be trusted.
    lo_pct, hi_pct = (float(x) for x in cfg["unit_robust_pct"])

    def p98_span(y):
        lo, hi = np.percentile(y, [lo_pct, hi_pct])
        return float(hi - lo)

    # Secondary, diagnostic only -- `distances` above stays the headline metric.
    # Spearman is invariant to ANY monotone rescaling, so it separates the two ways
    # a landscape can differ from MSE: a different SHAPE (rho drops) versus the same
    # shape mapped onto [0,1] differently because its global min/max sits elsewhere
    # in the volume (rho stays ~1 while the min-max distance grows). The pilot found
    # a subset where beta=0 had visibly MSE-like structure but a compressed range,
    # so the two can and do come apart.
    ranks = [float(spearmanr(f.ravel(), mse_u.ravel()).statistic) for f in feat_u]

    # Third variant of the same formula, differing only in the anchor. Stored so
    # the choice of headline metric never requires re-running the sweep.
    mse_r = unit_robust(mse, lo_pct, hi_pct)
    distances_robust = [
        float(np.mean((unit_robust(feat[i], lo_pct, hi_pct) - mse_r) ** 2))
        for i in range(len(encoder_records))
    ]

    diagnostics = {
        "distances_robust": distances_robust,
        "mse_p98_span": p98_span(mse_u),
        "feat_p98_span": [p98_span(f) for f in feat_u],
        "feat_spearman_vs_mse": ranks,
    }
    logger.info(f"mse p98_span={diagnostics['mse_p98_span']:.3f}")
    for r, d, dr, sp, rho in zip(
        encoder_records,
        distances,
        distances_robust,
        diagnostics["feat_p98_span"],
        ranks,
    ):
        logger.info(
            f"beta={r['beta']:g} train_seed={r['train_seed']}: "
            f"distance={d:.6g} distance_robust={dr:.6g} "
            f"p98_span={sp:.3f} spearman={rho:.4f}"
        )

    record = {
        "params": params,
        # The param_groups key for the same set, so a record joins against the
        # study ledger and the other score files without re-deriving it.
        "set_key": set_key,
        "seed": args.seed,
        "channel_type": args.channel_type,
        "opt_bounds": [float(x) for x in args.opt_bounds],
        "opt_samples": int(args.opt_samples),
        "unit_robust_pct": [lo_pct, hi_pct],
        "ideal_v_range": ideal_v_range,
        "backend": backend,
        "ideal_params": ideals,
        "encoders": encoder_records,
        "distances": distances,
        "diagnostics": diagnostics,
        "wall_s": round(time.time() - t0, 1),
    }
    _write_json(record_path, record)

    if args.save_losses:
        np.savez(
            out_dir / f"{stem}_losses.npz",
            mse=mse,
            feat=feat,
            betas=np.array([r["beta"] for r in encoder_records]),
            train_seeds=np.array([r["train_seed"] for r in encoder_records]),
            axis=axis,
        )

    logger.info(f"wrote {record_path} in {record['wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
