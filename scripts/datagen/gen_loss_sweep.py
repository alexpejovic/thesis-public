"""Simulate one N-D parameter grid once, score every loss on it, cache the result.

One sweep = one (parameter set, ideal-trace seed). It simulates the grid a single
time and evaluates ALL eight losses on those same traces, then scores all six
convexity descriptors on each of the eight landscapes. That is the whole reason
a cache is worth having: the 8 x 6 = 48 (loss, descriptor) cells that
``gen_convexity_scores.py`` asks for all come from one simulation, and the five
non-encoder losses cost ~14% on top of the three encoder ones.

Output is content-addressed, so a cache lookup is one ``Path.exists()`` -- no
directory scanning and no yaml parsing. (The first study looked runs up by
globbing ``data/comp_losses/*/config.yaml`` and parsing each one, which is O(n)
yaml reads per lookup and is exactly where a silent axis-transposition bug
lived.)

    data/loss_sweeps/<channel>/<key_hash>/
        manifest.json     the cache key, provenance, which losses are present
        descriptors.json  8 losses x 6 descriptors -- the cached product
        axis_samples.npy  (samples,) f64, the shared per-axis grid
        x_star.npy        (dim,) f64, the target, in manifest["params"] order
        losses/<name>.npy (samples,)*dim f64 -- only with --keep_grid
        log

Why descriptors and not grids
----------------------------
At 91 samples per axis a 3-D grid is 753,571 points, so the eight float64 loss
grids are 48 MB per sweep and the full study would be 269 GB. The 48 scored
floats are ~2 KB. ``--keep_grid`` retains the raw arrays for the handful of
sweeps you want to plot as landscapes; the trade-off is that re-scoring with a
different descriptor setting needs the sweep re-simulated. ``BASIN_WIDE_FRACTION``
and the strict-basin tolerance are therefore recorded in ``descriptors.json``,
so a later change to either is detectable rather than silent.

Usage (from encoding/):
    python scripts/datagen/gen_loss_sweep.py --params Na_gNa K_gK Km_gKm --seed 0
    python scripts/datagen/gen_loss_sweep.py --set_key Na_gNa+K_gK+Km_gKm --seed 0

`--set_key` is what `configs/studies/convexity_sweeps.yaml` drives, because a
study spec sends one token per flag and `--params` is variadic. `--no-generate`
reports the cache status without simulating, which is the only way to audit the
cache without a GPU.
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # encoding/
sys.path.insert(0, _HERE)  # for param_groups

import convexity as cx  # noqa: E402
import losses as L  # noqa: E402
import sim as S  # noqa: E402
from helper import get_datagen_config  # noqa: E402
from logger import get_logger, log_commit  # noqa: E402
from spike_metrics import cumulative_spike_count  # noqa: E402

SWEEP_SCHEMA = 1
OUT_BASE = Path("./data/loss_sweeps")

# A trace whose highest-amplitude sweep contains fewer than this many soft
# spikes is treated as non-spiking: every loss is then comparing two flat lines
# and the landscape carries no information.
MIN_SPIKES = 0.5

# `losses.LOSS_SET_VERSION` values whose shared loss definitions are identical,
# so a sweep cached under one is a valid sweep under another for every loss both
# sets contain. Version 2 is version 1 plus a single entry -- `encoder-zscore-2`,
# the same-config replicate encoder; the other eight `(kind, argument)` pairs and
# the `ENCODERS` datestrs they name are unchanged.
#
# This is load-bearing. `cache_state` used to call ANY version mismatch "stale",
# and `ensure_sweep` renames a stale sweep aside and re-simulates it -- so
# porting this module onto a tree at version 2 would have renamed and re-run all
# 5,605 sweeps cached on 2026-08-21, roughly 190 GPU-h, and orphaned the
# provenance of every number in `data/convexity-scores.jsonl`. Staleness is now
# decided per loss, by whether the descriptors hold what was actually asked for.
# Add a version here only after checking that the losses it shares with the
# others are defined identically.
COMPATIBLE_LOSS_SETS = frozenset({1, 2})


# ---------------------------------------------------------------------------
# cache key
# ---------------------------------------------------------------------------


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()[:16]


def sweep_key(channel_type, params, seed, bounds, samples, config) -> dict:
    """Everything that determines the simulated TRACES -- loss-independent.

    ``params`` is stored as an ordered LIST, never a mapping: a JSON object's
    key order is not meaningful and ``sort_keys=True`` would reorder it, which is
    precisely the bug that once transposed every multi-D landscape. A list is
    structurally immune.
    """
    return {
        "schema": SWEEP_SCHEMA,
        "channel_type": channel_type,
        "params": list(params),
        "seed": int(seed),
        "opt_bounds": [float(bounds[0]), float(bounds[1])],
        "opt_samples": int(samples),
        "sim": {
            "dt": config["dt"],
            "t_max": config["t_max"],
            "t_filter": config["t_filter"],
            "avg_scale": config["avg_scale"],
            "delay": config["current"]["delay"],
            "duration": config["current"]["duration"],
            "param_bounds_digest": _digest(config["param_bounds"][channel_type]),
            "param_consts_digest": _digest(config["param_consts"][channel_type]),
        },
    }


def key_hash(key: dict) -> str:
    return _digest(key)


def sweep_dir(key: dict) -> Path:
    return OUT_BASE / key["channel_type"] / key_hash(key)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _grid_axis(bounds, samples) -> np.ndarray:
    return np.linspace(float(bounds[0]), float(bounds[1]), int(samples))


def _batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def generate_sweep(
    channel_type,
    params,
    seed,
    bounds,
    samples,
    loss_names,
    *,
    config=None,
    batch_size=512,
    keep_grid=False,
    logger=None,
) -> Path:
    """Simulate the grid, evaluate every loss, score every descriptor, write."""
    config = config if config is not None else get_datagen_config()
    params = S.canonical(params)
    dim = len(params)
    key = sweep_key(channel_type, params, seed, bounds, samples, config)
    out = sweep_dir(key)
    out.mkdir(parents=True, exist_ok=True)

    log = logger if logger is not None else get_logger(out / "log", name="gen_loss_sweep")
    t0 = time.time()

    simulator = S.Sim(channel_type, config)
    x8 = simulator.targets_for_seeds([seed])[0]
    states6 = simulator.init_states_for_seeds([seed])[0]
    idx = S.param_indices(params)
    x_star = np.asarray(x8[idx], dtype=np.float64)

    ideal = simulator.traces_of(x8, states6)

    # Non-spiking check on the highest amplitude. Recorded rather than raised:
    # a silent neuron is a fact about that seed, not a crashed array task. The
    # legacy check needed the whole Allen feature extractor and raised.
    n_spikes = float(cumulative_spike_count(ideal[-1], simulator.dt_eff)[-1])
    spiking = n_spikes >= MIN_SPIKES
    log.info(f"seed={seed} params={params} ideal spikes={n_spikes:.2f}")

    manifest = {
        **key,
        "key_hash": key_hash(key),
        "n_amps": int(simulator.n_amps),
        "n_t": int(simulator.n_t),
        "spiking": bool(spiking),
        "n_spikes": n_spikes,
        "losses": [],
        "loss_set_version": L.LOSS_SET_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    np.save(out / "axis_samples.npy", _grid_axis(bounds, samples))
    np.save(out / "x_star.npy", x_star)

    if not spiking:
        log.info("ideal trace is non-spiking; recording and stopping")
        manifest["wall_s"] = round(time.time() - t0, 1)
        _write_json(out / "manifest.json", manifest)
        _write_json(out / "descriptors.json", {
            "loss_set_version": L.LOSS_SET_VERSION,
            "basin_wide_fraction": cx.BASIN_WIDE_FRACTION,
            "scores": {},
        })
        return out

    loss_fns = L.get_loss_fns(loss_names, channel_type, config)
    ordered = list(loss_fns)

    def _one_point(x_sub):
        full = x8.at[idx].set(x_sub)
        traces = simulator.traces_of(full, states6)
        return jnp.stack([loss_fns[n](traces, ideal) for n in ordered])

    eval_batch = jax.jit(jax.vmap(_one_point))

    axis = _grid_axis(bounds, samples)
    # itertools.product varies the LAST axis fastest, which is C order. Axis `i`
    # of the reshaped landscape is params[i]. Both facts are load-bearing and are
    # re-asserted on load.
    total = int(samples) ** dim
    chunks = []
    done = 0
    # Time-based progress, not count-based: a count interval tied to batch_size
    # logged once per 400k points at batch 4096, which is useless for watching a
    # long sweep. Every 30 s also makes the points/sec rate directly readable.
    last_log = time.time()
    for chunk in _batched(itertools.product(*[axis] * dim), batch_size):
        chunks.append(np.asarray(eval_batch(jnp.asarray(chunk))))
        done += len(chunk)
        now = time.time()
        if now - last_log >= 30.0 or done == total:
            rate = done / max(now - t0, 1e-9)
            eta = (total - done) / max(rate, 1e-9)
            log.info(f"{done}/{total} grid points "
                     f"({now - t0:.0f}s, {rate:.0f} pt/s, ETA {eta / 60:.1f} min)")
            last_log = now

    flat = np.concatenate(chunks, axis=0)  # (total, n_losses)
    if flat.shape != (total, len(ordered)):
        raise RuntimeError(f"grid shape {flat.shape} != {(total, len(ordered))}")

    grids = [axis] * dim
    scores = {}
    if keep_grid:
        (out / "losses").mkdir(exist_ok=True)
    for j, name in enumerate(ordered):
        y = flat[:, j].reshape((int(samples),) * dim, order="C")
        scores[name] = cx.score_all(y, grids, x_star)
        if keep_grid:
            np.save(out / "losses" / f"{name}.npy", y)

    descriptors = {
        "loss_set_version": L.LOSS_SET_VERSION,
        "basin_wide_fraction": cx.BASIN_WIDE_FRACTION,
        "basin_strict_tol": cx.basin_strict_tol(grids),
        "basin_wide_tol": cx.basin_wide_tol(grids),
        "scores": scores,
    }
    _merge_cached(out, manifest, descriptors, ordered, keep_grid, log)

    manifest["wall_s"] = round(time.time() - t0, 1)
    _write_json(out / "manifest.json", manifest)
    _write_json(out / "descriptors.json", descriptors)
    log.info(f"wrote {out} in {manifest['wall_s']}s")
    return out


def _merge_cached(out, manifest, descriptors, ordered, keep_grid, log) -> None:
    """Fold a topup into what the sweep already held, in place.

    `generate_sweep` is called with only the losses the caller asked for, so a
    plain write would replace an eight-loss `descriptors.json` with a one-loss one
    -- silently discarding seven scored landscapes that each cost a simulation.
    That was harmless while the only driver passed every loss; `--losses` makes it
    reachable, so merge: a newly scored loss wins, anything else is carried over,
    and `manifest["losses"]` becomes the union.

    Old scores are carried over ONLY if the descriptor settings match. `basin` and
    `basin_wide` are tolerance-dependent, so pooling scores computed under two
    different tolerances would produce a file whose rows are not comparable --
    exactly the kind of silent mixing `descriptors.json` records the tolerances to
    prevent.
    """
    old_desc_file = out / "descriptors.json"
    if not old_desc_file.exists():
        manifest["losses"] = list(ordered)
        manifest["kept_grid"] = bool(keep_grid)
        return
    try:
        old = json.loads(old_desc_file.read_text())
    except json.JSONDecodeError:
        log.info("existing descriptors.json is unreadable; replacing it")
        manifest["losses"] = list(ordered)
        manifest["kept_grid"] = bool(keep_grid)
        return

    tol_keys = ("basin_wide_fraction", "basin_strict_tol", "basin_wide_tol")
    changed = [k for k in tol_keys
               if k in old and old[k] != descriptors[k]]
    old_scores = old.get("scores", {})
    carried = [n for n in old_scores if n not in descriptors["scores"]]

    if changed and carried:
        log.info(f"descriptor settings changed ({', '.join(changed)}); dropping "
                 f"{len(carried)} cached score(s) rather than mixing tolerances: "
                 f"{sorted(carried)}")
        carried = []
    for name in carried:
        descriptors["scores"][name] = old_scores[name]
    if carried:
        log.info(f"carried {len(carried)} cached score(s) forward: "
                 f"{sorted(carried)}")

    manifest["losses"] = sorted(set(ordered) | set(carried))
    # Coarse by construction: the flag says "some raw grids are on disk", while
    # the grids themselves are per loss under `losses/`. Read the directory if you
    # need to know which.
    manifest["kept_grid"] = bool(keep_grid) or bool(
        carried and (out / "losses").is_dir()
    )


def _write_json(path: Path, obj) -> None:
    """Write atomically, so a reader never sees a half-written manifest."""
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def cache_state(channel_type, params, seed, bounds, samples, loss_names, config):
    """``(path, status)`` where status is 'hit', 'missing', 'topup' or 'stale'."""
    config = config if config is not None else get_datagen_config()
    key = sweep_key(channel_type, S.canonical(params), seed, bounds, samples, config)
    out = sweep_dir(key)
    man, desc = out / "manifest.json", out / "descriptors.json"
    if not (man.exists() and desc.exists()):
        return out, "missing"
    try:
        manifest = json.loads(man.read_text())
        descriptors = json.loads(desc.read_text())
    except json.JSONDecodeError:
        return out, "missing"

    # Not an equality test: see COMPATIBLE_LOSS_SETS. A version this module
    # knows to be loss-compatible is left alone, and the per-loss check below
    # decides whether anything is actually missing.
    if manifest.get("loss_set_version") not in COMPATIBLE_LOSS_SETS:
        return out, "stale"
    if not manifest.get("spiking", True):
        return out, "hit"  # a recorded non-spiking seed needs no re-run
    have = set(descriptors.get("scores", {}))
    if not set(loss_names) <= have:
        return out, "topup"
    return out, "hit"


def ensure_sweep(
    channel_type,
    params,
    seed,
    bounds,
    samples,
    loss_names,
    *,
    config=None,
    generate=True,
    batch_size=512,
    keep_grid=False,
    logger=None,
) -> tuple[Path, str]:
    """Return the sweep directory, simulating it only if the cache misses."""
    config = config if config is not None else get_datagen_config()
    out, status = cache_state(
        channel_type, params, seed, bounds, samples, loss_names, config
    )
    if status == "hit":
        return out, status
    if not generate:
        return out, status

    if status == "stale":
        # Never silently overwrite a landscape a published number may rest on.
        n = 1
        while (alt := out.with_name(out.name + f".v{n}")).exists():
            n += 1
        out.rename(alt)
        if logger:
            logger.info(f"loss_set_version changed; moved stale sweep to {alt}")

    generate_sweep(
        channel_type, params, seed, bounds, samples, loss_names,
        config=config, batch_size=batch_size, keep_grid=keep_grid, logger=logger,
    )
    return out, status


def read_descriptor(path: Path, loss_name: str, convexity: str):
    """One cached descriptor, or None if this seed was non-spiking."""
    descriptors = json.loads((path / "descriptors.json").read_text())
    scores = descriptors.get("scores", {})
    if loss_name not in scores:
        return None
    return float(scores[loss_name][convexity])


def load_grid(path: Path, loss_name: str):
    """``(y, grids, x_star)`` from a --keep_grid sweep, with order re-asserted."""
    manifest = json.loads((path / "manifest.json").read_text())
    y = np.load(path / "losses" / f"{loss_name}.npy")
    axis = np.load(path / "axis_samples.npy")
    x_star = np.load(path / "x_star.npy")

    params = manifest["params"]
    dim = len(params)
    if params != S.canonical(params):
        raise RuntimeError(f"{path}: stored params {params} are not canonical")
    if y.ndim != dim or y.shape != (len(axis),) * dim:
        raise RuntimeError(f"{path}: grid shape {y.shape} != {(len(axis),) * dim}")
    if x_star.shape != (dim,):
        raise RuntimeError(f"{path}: x_star shape {x_star.shape} != ({dim},)")
    return y, [axis] * dim, x_star


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_losses(text: str) -> list[str]:
    """Comma-separated, because a study spec sends one token per flag.

    `study.argv_for` renders every flag value with `str(v)`, so `nargs="+"` can
    never be driven from a spec -- the same reason `configs/studies/*.yaml` must
    not carry a list-valued key. Validated here so a typo is an argparse error
    before the simulator is built, not a KeyError after it.
    """
    names = [n for n in (t.strip() for t in text.split(",")) if n]
    if not names:
        raise argparse.ArgumentTypeError("--losses is empty")
    unknown = [n for n in names if n not in L.LOSS_NAMES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown loss(es) {unknown}; known: {list(L.LOSS_NAMES)}"
        )
    seen, out = set(), []
    for n in names:  # order-preserving dedup; the loss order sets column order
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def main(argv=None) -> int:
    import param_groups as pg

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set_key", help="parameter set as a param_groups key, "
                                      "e.g. Na_gNa+K_gK+Km_gKm")
    ap.add_argument("--params", nargs="+",
                    help="parameters, instead of --set_key (interactive use: "
                         "a study spec sends one token per flag, so it must use "
                         "--set_key)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--channel_type", default="Pospischil")
    ap.add_argument("--opt_bounds", nargs=2, type=float, default=None)
    ap.add_argument("--opt_samples", type=int, default=None)
    ap.add_argument("--losses", type=parse_losses,
                    default=list(L.LOSS_NAMES),
                    help="comma-separated subset of the loss set; default all "
                         f"{len(L.LOSS_NAMES)} of them. NOTE that asking for a "
                         "loss a cached sweep does not hold forces a full "
                         "re-simulation -- the raw grids are not kept, so a new "
                         "loss cannot be scored from cache.")
    ap.add_argument("--batch_size", type=int, default=512,
                    help="grid points per jitted call; raise on GPU")
    ap.add_argument("--keep_grid", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="also store the raw loss grids (48 MB/sweep at 91^3)")
    ap.add_argument("--generate", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="simulate a missing or incomplete sweep (default). "
                         "--no-generate reports the cache status and exits 0, "
                         "which is how to audit the cache without a GPU.")
    ap.add_argument("--force", action="store_true", help="ignore a cache hit")
    args = ap.parse_args(argv)

    if (args.set_key is None) == (args.params is None):
        ap.error("give exactly one of --set_key or --params")

    config = get_datagen_config()
    params = S.canonical(args.params if args.params else pg.params_for(args.set_key))
    key = pg.set_key(params)
    # Only consult the grid plan for values not given explicitly, so a smaller
    # test sweep (or a dimensionality the plan does not cover) still runs.
    if args.opt_bounds is None or args.opt_samples is None:
        grid = pg.grid_for(key)
        bounds = args.opt_bounds if args.opt_bounds is not None else grid["bounds"]
        samples = args.opt_samples if args.opt_samples is not None else grid["samples"]
    else:
        bounds, samples = args.opt_bounds, args.opt_samples

    out, status = cache_state(
        args.channel_type, params, args.seed, bounds, samples, args.losses, config
    )
    if status == "hit" and not args.force:
        print(f"cache hit: {out}")
        return 0
    if not args.generate:
        # Reporting IS the work in this mode, so this is a success. Anything that
        # stops the requested work from happening still has to exit non-zero.
        print(f"{status}: {out}  set={key} seed={args.seed}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    logger = get_logger(out / "log", name="gen_loss_sweep")
    log_commit(logger, "../")
    logger.info(f"set={key} params={params} seed={args.seed} "
                f"bounds={tuple(bounds)} samples={samples} status={status} "
                f"losses={args.losses}")

    generate_sweep(
        args.channel_type, params, args.seed, bounds, samples, args.losses,
        config=config, batch_size=args.batch_size, keep_grid=args.keep_grid,
        logger=logger,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
