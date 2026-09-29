"""Generic SLURM study orchestration: turn a YAML spec into a job array.

A *study* is a cartesian product of parameter axes run against one existing
entry point. The spec lives in ``./configs/studies/<name>.yaml`` and names the
target script, the axes to sweep, the flags to hold fixed, the resources one
work unit needs, and a ledger file for resume. Nothing here knows anything about
losses, encoders or convexity -- adding a study means adding a YAML file, not
editing this module.

This generalizes the hand-written fan-out scripts (``mass_gen_losses.sh`` and
friends), which hardcode their value list in bash and re-submit everything on
every invocation. The three things it adds are the reason it exists:

1. **One frozen manifest per submission.** ``--materialize`` writes the pending
   job list to ``./data/studies/<name>/<datestr>/jobs.jsonl`` and array tasks
   index into *that*, never into a freshly recomputed list. Recomputing per task
   is a real bug: the ledger grows while the array runs, so task 7 would resolve
   a different job than the submitter counted, and jobs would be skipped or run
   twice.
2. **Resume by ledger.** A work unit's key is looked up in an append-only JSONL
   ledger, so a study that dies at the wall clock is re-submitted with only the
   missing cells. The target script does not need to know it is resumable.
3. **Validation up front.** The flag surface of every target is auto-generated
   from its own YAML config (see ``_parse_args`` in the datagen scripts), so a
   misspelled axis name is not an error until it reaches argparse -- in every one
   of several hundred tasks at once. ``validate`` catches it before submission.

Spec keys
---------
``script``       target, relative to ``encoding/`` (required)
``base_config``  the YAML the target reads; used to validate flag names
``axes``         ordered mapping name -> values; the product is the job list
``flags``        fixed flags added to every job
``key``          ledger key field names (default: every axis name)
``resources``    ``partition``, ``gres``, ``cpus_per_task``, ``mem``,
                 ``time_per_unit``; the submitter multiplies time by the chunk
``chunk``        work units per array task (null = derive from ``max_submit``)
``cap``          max simultaneously running array tasks
``ledger``       append-only JSONL recording finished work units
``inputs``       paths the preflight requires to exist

An axis value is a plain list, ``{range: [lo, hi]}`` / ``{range: [lo, hi,
step]}`` (half-open, like ``range``), or ``{from_file: path}`` (one value per
line, ``#`` comments ignored) so a long list of datestrs need not be inlined.

CLI, all of it consumed by the two shell scripts in ``slurm_scripts/``::

    python study.py --study loss_sweeps --plan
    python study.py --study loss_sweeps --materialize
    python study.py --manifest <dir> --count
    python study.py --manifest <dir> --index 3 --argv
    python study.py --manifest <dir> --index 3 --record --status ok
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jsonl_store as JS  # noqa: E402

STUDY_DIR = Path("./configs/studies")
MANIFEST_ROOT = Path("./data/studies")
DEFAULT_LEDGER = Path("./data/study-runs.jsonl")

# Ledger fields that identify a work unit. The axis values are stored under
# "key" as a list, so `study` + `cell` is what dedup actually compares.
LEDGER_FIELDS = ("study", "cell")

DEFAULT_RESOURCES = {
    "partition": "a100-galvani",
    "gres": "gpu:1",
    "cpus_per_task": 20,
    "mem": "50G",
    "time_per_unit": "0-00:30",
}


# ---------------------------------------------------------------------------
# Spec loading and validation
# ---------------------------------------------------------------------------


def load_study(name_or_path: str) -> dict:
    """Load a study spec by name (in ``configs/studies/``) or by path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = STUDY_DIR / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"no study spec at {path}. Available: "
            f"{', '.join(sorted(p.stem for p in STUDY_DIR.glob('*.yaml'))) or 'none'}"
        )
    spec = yaml.safe_load(path.read_text()) or {}
    spec.setdefault("name", path.stem)
    spec.setdefault("axes", {})
    spec.setdefault("flags", {})
    spec.setdefault("inputs", [])
    spec.setdefault("ledger", str(DEFAULT_LEDGER))
    spec.setdefault("cap", 8)
    spec.setdefault("chunk", None)
    spec["resources"] = {**DEFAULT_RESOURCES, **(spec.get("resources") or {})}
    spec.setdefault("key", list(spec["axes"]))
    spec["_path"] = str(path)
    return spec


def _axis_values(name: str, raw: Any) -> list:
    """Expand one axis definition into an explicit list of values."""
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, dict) and "range" in raw:
        args = raw["range"]
        if not 2 <= len(args) <= 3:
            raise ValueError(f"axis {name!r}: range takes [lo, hi] or [lo, hi, step]")
        values = list(range(*args))
    elif isinstance(raw, dict) and "from_file" in raw:
        text = Path(raw["from_file"]).read_text()
        values = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        raise ValueError(
            f"axis {name!r}: expected a list, {{range: [...]}} or "
            f"{{from_file: path}}, got {type(raw).__name__}"
        )
    if not values:
        raise ValueError(f"axis {name!r} expanded to no values")
    return values


def axes_of(spec: dict) -> dict[str, list]:
    return {name: _axis_values(name, raw) for name, raw in spec["axes"].items()}


def all_jobs(spec: dict) -> list[dict]:
    """The full cartesian product, in spec order. Axis order fixes job order."""
    axes = axes_of(spec)
    if not axes:
        return [{}]
    names = list(axes)
    return [dict(zip(names, combo)) for combo in product(*(axes[n] for n in names))]


def cell(spec: dict, job: dict) -> str:
    """Stable label for one work unit: the key fields joined with ``|``.

    Used verbatim in the ledger and in log lines, so renaming a key field
    orphans the existing ledger rows for that study.
    """
    return "|".join(f"{k}={job[k]}" for k in spec["key"] if k in job)


def argv_for(spec: dict, job: dict) -> list[str]:
    """The CLI tokens for one work unit: fixed flags first, then axis values."""
    tokens: list[str] = []
    for source in (spec["flags"], job):
        for k, v in source.items():
            if isinstance(v, bool):
                tokens.append(f"--{k}" if v else f"--no-{k}")
            else:
                tokens += [f"--{k}", str(v)]
    return tokens


def default_of(spec: dict, name: str):
    """The first value this spec supplies for ``name`` (axis or fixed flag)."""
    if name in spec["flags"]:
        return spec["flags"][name]
    return _axis_values(name, spec["axes"][name])[0]


def validate(spec: dict) -> list[str]:
    """Every problem that would otherwise surface inside an array task.

    Returned as a list rather than raised so the preflight can print all of
    them at once instead of one per re-run.
    """
    problems: list[str] = []

    script = spec.get("script")
    if not script:
        problems.append("spec has no 'script' key")
    elif not Path(script).is_file():
        problems.append(f"script not found: {script}")

    try:
        jobs = all_jobs(spec)
    except (ValueError, OSError) as exc:
        return problems + [f"axes do not expand: {exc}"]
    if not jobs:
        problems.append("the axes expand to no jobs")

    # A value containing a newline would break the one-token-per-line argv
    # protocol the shell side reads with `mapfile`.
    for job in jobs:
        for k, v in job.items():
            if "\n" in str(v):
                problems.append(f"axis {k!r} has a value containing a newline")
                break

    base = spec.get("base_config")
    if base and Path(base).is_file():
        config = yaml.safe_load(Path(base).read_text()) or {}
        named = list(spec["axes"]) + list(spec["flags"])
        for k in named:
            if k not in config:
                problems.append(
                    f"{k!r} is not a key in {base}, so --{k} does not exist "
                    f"(flags are auto-generated from that YAML)"
                )
            elif isinstance(config[k], list):
                # _parse_args uses type=type(v), so a list-valued key becomes
                # type=list and argparse shreds "-1.5 1.5" into single
                # characters. Silent and very hard to spot in a log.
                problems.append(
                    f"{k!r} is list-valued in {base}; argparse would parse "
                    f"--{k} character by character. Change the default to a "
                    f"scalar or leave this key alone."
                )
            elif type(config[k]) is not type(default_of(spec, k)) and not (
                isinstance(config[k], float) and isinstance(default_of(spec, k), int)
            ):
                # argparse uses type=type(v) from the base config, so a value of
                # the wrong type is either rejected or silently coerced. The
                # common cause is YAML 1.1: an unquoted 2026_08_12_15_20_52_343885
                # is an INTEGER with underscore digit separators, which turns
                # every datestr into 20260812152052343885 with no error anywhere.
                problems.append(
                    f"{k!r} is {type(config[k]).__name__} in {base} but this "
                    f"spec supplies {type(default_of(spec, k)).__name__} "
                    f"({default_of(spec, k)!r}). If that should be a string, "
                    f"quote it -- YAML reads 2026_08_12_… as an int."
                )
    elif base:
        problems.append(f"base_config not found: {base}")

    if not isinstance(spec["cap"], int) or spec["cap"] < 1:
        problems.append(f"cap must be a positive int, got {spec['cap']!r}")
    if spec["chunk"] is not None and (
        not isinstance(spec["chunk"], int) or spec["chunk"] < 1
    ):
        problems.append(f"chunk must be a positive int or null, got {spec['chunk']!r}")

    try:
        parse_duration(spec["resources"]["time_per_unit"])
    except ValueError as exc:
        problems.append(str(exc))

    return problems


# ---------------------------------------------------------------------------
# Wall clock
# ---------------------------------------------------------------------------


def parse_duration(text: str) -> int:
    """Seconds from a SLURM duration: ``D-HH:MM:SS``, ``D-HH:MM``, ``HH:MM:SS``,
    ``MM:SS`` or bare minutes."""
    text = str(text).strip()
    days = 0
    # Whether the "D-" prefix was PRESENT, not whether it was nonzero: SLURM
    # reads "0-00:30" as HH:MM (30 minutes) but a bare "00:30" as MM:SS.
    had_days = "-" in text
    if had_days:
        d, _, text = text.partition("-")
        try:
            days = int(d)
        except ValueError:
            raise ValueError(f"bad duration {text!r}: {d!r} is not a day count")
    parts = text.split(":") if text else ["0"]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"bad duration {text!r}")
    if len(nums) == 1:
        h, m, s = 0, nums[0], 0
    elif len(nums) == 2:
        h, m, s = (nums[0], nums[1], 0) if had_days else (0, nums[0], nums[1])
    elif len(nums) == 3:
        h, m, s = nums
    else:
        raise ValueError(f"bad duration {text!r}: too many ':' fields")
    return days * 86400 + h * 3600 + m * 60 + s


def format_duration(seconds: int) -> str:
    """SLURM ``D-HH:MM:SS``."""
    seconds = max(int(seconds), 60)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Ledger and resume
# ---------------------------------------------------------------------------


def done_cells(spec: dict) -> set[str]:
    """Cells already recorded ``ok`` for this study."""
    rows = JS.load_rows(spec["ledger"])
    return {
        r["cell"]
        for r in rows
        if r.get("study") == spec["name"] and r.get("status") == "ok"
    }


def pending_jobs(spec: dict, *, resume: bool = True) -> list[dict]:
    jobs = all_jobs(spec)
    if not resume:
        return jobs
    done = done_cells(spec)
    return [j for j in jobs if cell(spec, j) not in done]


def record(spec: dict, job: dict, status: str, elapsed_s: float | None = None) -> None:
    JS.append_row(
        spec["ledger"],
        {
            "study": spec["name"],
            "cell": cell(spec, job),
            "key": [job.get(k) for k in spec["key"]],
            "status": status,
            "elapsed_s": None if elapsed_s is None else round(float(elapsed_s), 1),
            "recorded": datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    )


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


def materialize(spec: dict, *, resume: bool = True, limit: int | None = None) -> Path:
    """Freeze the pending job list so array tasks index a fixed list.

    Returns the manifest directory. ``jobs.jsonl`` is one job per line and the
    line number *is* the array task's job index.
    """
    jobs = pending_jobs(spec, resume=resume)
    if limit is not None:
        jobs = jobs[:limit]
    stamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f")
    out = MANIFEST_ROOT / spec["name"] / stamp
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "jobs.jsonl", "w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "study": spec["name"],
                "spec_path": spec["_path"],
                "created": stamp,
                "resume": resume,
                "n_total": len(all_jobs(spec)),
                "n_jobs": len(jobs),
                "spec": {k: v for k, v in spec.items() if not k.startswith("_")},
            },
            indent=2,
        )
        + "\n"
    )
    return out


def load_manifest(path: str | Path) -> tuple[dict, list[dict]]:
    """The spec snapshot and the frozen job list from a manifest directory."""
    d = Path(path)
    if d.is_file():  # tolerate being handed manifest.json itself
        d = d.parent
    meta = json.loads((d / "manifest.json").read_text())
    spec = meta["spec"]
    spec["_path"] = meta.get("spec_path", str(d))
    jobs = [
        json.loads(line)
        for line in (d / "jobs.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return spec, jobs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def chunk_for(spec: dict, n_jobs: int, max_submit: int) -> int:
    if spec["chunk"]:
        return int(spec["chunk"])
    return max(1, -(-n_jobs // max_submit))  # ceil


def plan_text(spec: dict, max_submit: int) -> str:
    axes = axes_of(spec)
    total = len(all_jobs(spec))
    pending = len(pending_jobs(spec))
    chunk = chunk_for(spec, pending, max_submit)
    tasks = max(1, -(-pending // chunk))
    per_unit = parse_duration(spec["resources"]["time_per_unit"])

    lines = [f"study        {spec['name']}   ({spec['_path']})"]
    if spec.get("description"):
        lines.append(f"             {spec['description']}")
    lines.append(f"script       {spec['script']}")
    for name, values in axes.items():
        preview = ", ".join(str(v) for v in values[:4])
        more = f", ... (+{len(values) - 4})" if len(values) > 4 else ""
        lines.append(f"axis         {name} [{len(values)}]: {preview}{more}")
    if spec["flags"]:
        fixed = " ".join(f"--{k} {v}" for k, v in spec["flags"].items())
        lines.append(f"fixed        {fixed}")
    lines += [
        f"key          {'|'.join(spec['key'])}",
        f"ledger       {spec['ledger']}",
        "",
        f"units        {total} total, {total - pending} done, {pending} pending",
        f"chunk        {chunk} unit(s)/task  ->  {tasks} array task(s), "
        f"cap {spec['cap']}",
        f"wall clock   {spec['resources']['time_per_unit']}/unit  ->  "
        f"{format_duration(per_unit * chunk)}/task",
        f"resources    {spec['resources']['partition']} "
        f"gres={spec['resources']['gres']} "
        f"cpus={spec['resources']['cpus_per_task']} "
        f"mem={spec['resources']['mem']}",
    ]
    return "\n".join(lines)


def sbatch_args(spec: dict, chunk: int, max_wall_s: int | None = None) -> list[str]:
    r = spec["resources"]
    wall = parse_duration(r["time_per_unit"]) * chunk
    if max_wall_s is not None:
        wall = min(wall, max_wall_s)
    args = [
        f"--partition={r['partition']}",
        f"--cpus-per-task={r['cpus_per_task']}",
        f"--mem={r['mem']}",
        f"--time={format_duration(wall)}",
        f"--job-name={spec['name']}",
        f"--output=logs/%A_%a-{spec['name']}.out",
        f"--error=logs/%A_%a-{spec['name']}.err",
    ]
    if r.get("gres"):
        args.append(f"--gres={r['gres']}")
    return args


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--study", help="spec name in ./configs/studies, or a path")
    src.add_argument("--manifest", help="a materialized manifest directory")

    ap.add_argument("--index", type=int, help="job index within the manifest")
    ap.add_argument("--count", action="store_true", help="print the job count")
    ap.add_argument("--plan", action="store_true", help="print the work plan")
    ap.add_argument("--validate", action="store_true", help="print problems, if any")
    ap.add_argument("--inputs", action="store_true", help="print preflight paths")
    ap.add_argument("--dump", action="store_true", help="TSV of every job")
    ap.add_argument("--argv", action="store_true",
                    help="print the target's CLI tokens, one per line")
    ap.add_argument("--field", help="print one field: script, cell, ledger, name")
    ap.add_argument("--sbatch-args", action="store_true",
                    help="print sbatch resource flags, one per line")
    ap.add_argument("--materialize", action="store_true",
                    help="freeze the pending job list; prints the manifest dir")
    ap.add_argument("--record", action="store_true",
                    help="append a ledger row for --index")
    ap.add_argument("--status", default="ok", help="status for --record")
    ap.add_argument("--elapsed", type=float, help="seconds for --record")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the ledger; treat every unit as pending")
    ap.add_argument("--limit", type=int, help="materialize at most N units (pilot)")
    ap.add_argument("--max-submit", type=int, default=300,
                    help="QOS MaxSubmitJobs, used to derive the chunk")
    ap.add_argument("--chunk", type=int, help="override the spec's chunk")
    ap.add_argument("--max-wall-h", type=int,
                    help="clamp the derived --time to this many hours")
    args = ap.parse_args(argv)

    if args.manifest:
        spec, jobs = load_manifest(args.manifest)
    else:
        spec = load_study(args.study)
        jobs = pending_jobs(spec, resume=not args.no_resume)
    if args.chunk:
        spec["chunk"] = args.chunk

    if args.validate:
        problems = validate(spec)
        for p in problems:
            print(f"  {p}")
        return 1 if problems else 0

    if args.plan:
        print(plan_text(spec, args.max_submit))
        return 0

    if args.inputs:
        for p in spec["inputs"]:
            print(p)
        return 0

    if args.materialize:
        print(materialize(spec, resume=not args.no_resume, limit=args.limit))
        return 0

    if args.sbatch_args:
        chunk = args.chunk or chunk_for(spec, len(jobs), args.max_submit)
        max_wall = args.max_wall_h * 3600 if args.max_wall_h else None
        for a in sbatch_args(spec, chunk, max_wall):
            print(a)
        return 0

    # A spec-level field needs no index; only "cell" is per-unit.
    if args.field and args.field != "cell" and args.index is None:
        print(spec.get(args.field, ""))
        return 0

    if args.dump:
        for i, job in enumerate(jobs):
            print(f"{i}\t{cell(spec, job)}\t{' '.join(argv_for(spec, job))}")
        return 0

    if args.count:
        print(len(jobs))
        return 0

    if args.index is None:
        ap.error("--index is required unless --count/--plan/--dump/--materialize")
    if not 0 <= args.index < len(jobs):
        print(f"index {args.index} out of range 0..{len(jobs) - 1}", file=sys.stderr)
        return 1
    job = jobs[args.index]

    if args.record:
        record(spec, job, args.status, args.elapsed)
        return 0

    if args.argv:
        # One token per line so the shell can `mapfile` it. This is why values
        # may not contain newlines, and why no quoting or `eval` is involved.
        for token in argv_for(spec, job):
            print(token)
        return 0

    if args.field:
        print({"cell": cell(spec, job), **spec}.get(args.field, ""))
        return 0

    print(cell(spec, job))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
