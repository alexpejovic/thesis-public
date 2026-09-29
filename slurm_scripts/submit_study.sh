#!/usr/bin/env bash
# Preflight and submit any study declared in encoding/configs/studies/.
#
#   ./submit_study.sh <study> --dry-run   # preflight + work plan, no submission
#   ./submit_study.sh <study> --pilot     # a couple of units, to measure cost
#   ./submit_study.sh <study>             # the whole study (resumes by default)
#   ./submit_study.sh --list              # what studies exist
#
# Options:
#   --dry-run     preflight and print the plan; submit nothing
#   --pilot       submit PILOT_UNITS units at chunk 1, so a real per-unit wall
#                 clock can be measured before committing the full array
#   --skip-pilot  submit the full study without a pilot on record
#   --no-resume   ignore the ledger and re-run every unit
#   --chunk N     units per array task, overriding the spec
#   --cap N       max simultaneously running tasks, overriding the spec
#
# Two stages on purpose: a study's cost is dominated by one number nobody has
# measured yet -- the per-unit wall clock -- so the full submit refuses until a
# pilot has run, unless --skip-pilot is given.
set -euo pipefail
cd "$(dirname "$0")"

# QOS limits, verified with:
#   sacctmgr show qos format=Name,MaxSubmitJobsPerUser,MaxJobsPerUser,MaxWall
# normal/gpunormal: MaxSubmit 300, MaxJobs 36, MaxWall 3-00:00:00.
MAX_SUBMIT=${MAX_SUBMIT:-300}
MAX_RUNNING=${MAX_RUNNING:-36}
MAX_WALL_H=${MAX_WALL_H:-72}
PILOT_UNITS=${PILOT_UNITS:-2}

# export, not a bare assignment: --export=ALL,PY only propagates an ENVIRONMENT
# variable, and a silent fallback to an empty venv makes every array task exit 0
# having done nothing.
#
# Resolved to an ABSOLUTE path: this script runs from slurm_scripts/ but the
# array task runs from encoding/, so any relative interpreter path would mean
# two different files in the two places.
PY=${PY:-$(cd ../.. && pwd)/.venv/bin/python}
# -ms, not -m: .venv/bin/python is a SYMLINK to the base conda interpreter, and
# resolving it breaks site-packages -- the venv must be invoked by its own path.
export PY=$(realpath -ms "$PY")

STUDY=""
DRY_RUN=0; PILOT=0; SKIP_PILOT=0; NO_RESUME=0
CHUNK_OVERRIDE=""; CAP_OVERRIDE=""
while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --pilot) PILOT=1 ;;
        --skip-pilot) SKIP_PILOT=1 ;;
        --no-resume) NO_RESUME=1 ;;
        --chunk) CHUNK_OVERRIDE=$2; shift ;;
        --cap) CAP_OVERRIDE=$2; shift ;;
        --list)
            echo "studies in encoding/configs/studies:"
            for f in ../configs/studies/*.yaml; do
                [[ -e "$f" ]] || { echo "  (none)"; break; }
                printf '  %-24s %s\n' "$(basename "$f" .yaml)" \
                    "$(sed -n 's/^description: *//p' "$f" | head -1)"
            done
            exit 0 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) STUDY=$1 ;;
    esac
    shift
done

if [[ -z "$STUDY" ]]; then
    echo "usage: ./submit_study.sh <study> [--dry-run|--pilot]" >&2
    echo "       ./submit_study.sh --list" >&2
    exit 2
fi

RESUME_FLAG=()
((NO_RESUME)) && RESUME_FLAG=(--no-resume)

# study.py resolves ./configs and ./data relative to encoding/, like every other
# script in this repo.
py_study() { (cd .. && "$PY" study.py --study "$STUDY" "$@"); }

fail=0
note_missing() { printf '  MISSING  %s\n' "$1"; fail=1; }
need() {
    if [[ -e "../$1" || -e "$1" ]]; then printf '  ok       %s\n' "$1"
    else note_missing "$1"; fi
}

echo "== preflight =="
if [[ -x "$PY" ]]; then
    printf '  ok       %s\n' "$PY"
    if "$PY" -c "import jax, jaxley, optax, yaml" 2>/dev/null; then
        echo "  ok       python deps importable"
    else
        echo "  MISSING  deps in $PY (need jax, jaxley, optax, pyyaml)"
        echo "           run 'make requirements' at the repo root, or set"
        echo "           PY=/path/to/a/populated/.venv/bin/python"
        fail=1
    fi
else
    note_missing "$PY (not executable)"
fi

# The spec must be valid before submission, not inside 300 tasks at once: the
# target's flags are auto-generated from its own YAML, so a misspelled or
# list-valued name is an argparse error in every single task.
if [[ -x "$PY" ]]; then
    if problems=$(py_study --validate); then
        echo "  ok       spec validates"
    else
        echo "  MISSING  spec problems:"
        echo "$problems"
        fail=1
    fi
    need "$(py_study --field script 2>/dev/null || echo '<unknown script>')"
    while read -r p; do
        [[ -n "$p" ]] && need "$p"
    done < <(py_study --inputs 2>/dev/null || true)
fi

# flock must be honoured on the shared data dir, or concurrent tasks on
# different nodes corrupt the ledger. A failure here is not fatal: jsonl_store
# falls back to one file per row, which load_rows reads transparently.
if [[ -x "$PY" ]]; then
    if (cd .. && "$PY" -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
import jsonl_store as JS
sys.exit(0 if JS.probe_flock(Path('./data/.flock_probe.jsonl')) else 1)
") 2>/dev/null; then
        echo "  ok       flock honoured on ./data"
        rm -f ../data/.flock_probe.jsonl.lock
    else
        echo "  WARNING  flock NOT honoured on ./data -- the per-row shard"
        echo "           fallback will be used (correct, just more files)"
    fi
fi

mkdir -p logs
if ((fail)); then echo -e "\npreflight FAILED; not submitting." >&2; exit 1; fi

echo -e "\n== work plan =="
py_study --plan --max-submit "$MAX_SUBMIT" "${RESUME_FLAG[@]}"

N_UNITS=$(py_study --count "${RESUME_FLAG[@]}")
if ((N_UNITS == 0)); then
    echo -e "\nnothing pending: every unit is already in the ledger."
    echo "Pass --no-resume to re-run them anyway."
    exit 0
fi

if ((PILOT)); then
    ((N_UNITS > PILOT_UNITS)) && N_UNITS=$PILOT_UNITS
    CHUNK=1
    CAP=$((PILOT_UNITS < 4 ? PILOT_UNITS : 4))
    LIMIT_FLAG=(--limit "$PILOT_UNITS")
else
    CHUNK=${CHUNK_OVERRIDE:-$(py_study --field chunk)}
    [[ -z "$CHUNK" || "$CHUNK" == "None" ]] && CHUNK=$(( (N_UNITS + MAX_SUBMIT - 1) / MAX_SUBMIT ))
    ((CHUNK < 1)) && CHUNK=1
    CAP=${CAP_OVERRIDE:-$(py_study --field cap)}
    LIMIT_FLAG=()
fi
N_TASKS=$(( (N_UNITS + CHUNK - 1) / CHUNK ))

echo -e "\n== submission =="
printf '%-14s %8s %7s %7s %6s\n' study units chunk tasks cap
printf '%-14s %8s %7s %7s %6s\n' "$STUDY" "$N_UNITS" "$CHUNK" "$N_TASKS" "$CAP"

# Guards. MaxSubmit counts jobs ALREADY queued, so check against the live count
# rather than against an empty queue.
QUEUED=$(squeue -h -u "$USER" -o '%i' 2>/dev/null | wc -l || echo 0)
if ((N_TASKS + QUEUED > MAX_SUBMIT)); then
    echo "refusing: $N_TASKS tasks + $QUEUED already queued > MaxSubmit=$MAX_SUBMIT." >&2
    echo "          raise --chunk to shrink the array." >&2
    exit 1
fi
if ((CAP > MAX_RUNNING)); then
    echo "refusing: cap $CAP > MaxJobs=$MAX_RUNNING." >&2
    exit 1
fi

# A chunk whose total exceeds MaxWall gets its --time clamped, which means the
# last units in every task are killed at the wall clock. Warn rather than
# refuse: the per-unit estimate in the spec is usually pessimistic, and the
# ledger makes a re-submission cheap.
UNIT_S=$(cd .. && "$PY" -c "
import sys; sys.path.insert(0, '.')
import study; print(study.parse_duration(
    study.load_study('$STUDY')['resources']['time_per_unit']))")
if ((UNIT_S * CHUNK > MAX_WALL_H * 3600)); then
    echo "WARNING: $CHUNK units x ${UNIT_S}s = $((UNIT_S * CHUNK / 3600))h exceeds" >&2
    echo "         MaxWall=${MAX_WALL_H}h, so --time is clamped and the last" >&2
    echo "         units per task will hit the wall clock. Lower --chunk, or" >&2
    echo "         re-run afterwards: the ledger resumes what did not finish." >&2
fi

mapfile -t SB < <(py_study --sbatch-args --chunk "$CHUNK" --max-wall-h "$MAX_WALL_H")
SBATCH_CMD=(sbatch --parsable "${SB[@]}"
    --export=ALL,PY,STUDY_MANIFEST,CHUNK
    --array=0-$((N_TASKS - 1))%"$CAP"
    study_array.sh)

if ((DRY_RUN)); then
    echo -e "\nwould run:"
    printf '  STUDY_MANIFEST=<materialized at submit> CHUNK=%s \\\n' "$CHUNK"
    printf '  %s \\\n' "${SBATCH_CMD[@]:0:${#SBATCH_CMD[@]}-1}"
    printf '  %s\n' "${SBATCH_CMD[-1]}"
    echo -e "\n--dry-run: nothing submitted."
    exit 0
fi

if ((PILOT == 0 && SKIP_PILOT == 0)); then
    LEDGER=$(py_study --field ledger)
    if [[ ! -s "../${LEDGER#./}" ]]; then
        echo -e "\nrefusing: no pilot result in $LEDGER." >&2
        echo "Run './submit_study.sh $STUDY --pilot' first -- it measures the" >&2
        echo "per-unit wall clock the whole plan hangs on -- or pass" >&2
        echo "--skip-pilot to override." >&2
        exit 1
    fi
fi

# Freeze the pending unit list. Array tasks index into THIS file, so the ledger
# filling up mid-run cannot shift what task N means.
MANIFEST=$(py_study --materialize "${RESUME_FLAG[@]}" "${LIMIT_FLAG[@]}")
# Absolute: study.py prints it relative to encoding/, and the array task runs
# there, but an absolute path cannot be misread if that ever changes.
export STUDY_MANIFEST=$(cd .. && realpath -ms "$MANIFEST")
export CHUNK
echo "manifest      $MANIFEST"

JOB_ID=$("${SBATCH_CMD[@]}")

echo "submitted     job $JOB_ID  ($N_TASKS tasks x $CHUNK unit(s), cap $CAP)"

cat <<EOF

Watch:     squeue -j $JOB_ID
Failures:  sacct -j $JOB_ID --state=FAILED,TIMEOUT --format=JobID,State,Elapsed
Logs:      logs/${JOB_ID}_<task>-${STUDY}.{out,err}
Progress:  ./submit_study.sh $STUDY --dry-run     # re-reads the ledger

Results are recorded per unit, so re-running the same command after a timeout
submits only what is still missing.
EOF
