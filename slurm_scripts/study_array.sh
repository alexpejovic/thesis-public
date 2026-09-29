#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --nodes=1
#SBATCH --time=0-01:00
#SBATCH --partition=a100-galvani
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --output=logs/%A_%a-study.out
#SBATCH --error=logs/%A_%a-study.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alex.pejovic@student.uni-tuebingen.de
#
# ONE array body for every study. The #SBATCH values above are only fallbacks --
# submit_study.sh passes the spec's resources (and per-study log names) as sbatch
# CLI flags, which override this header.
#
# Do not sbatch this by hand; it needs STUDY_MANIFEST from the submitter:
#
#   ./submit_study.sh <study>
#
# Each task runs CHUNK work units in sequence, because the QOS caps a submission
# at 300 array tasks and a study can be larger than that. The unit index is
#
#     STUDY_MANIFEST/jobs.jsonl line (SLURM_ARRAY_TASK_ID * CHUNK + k)
#
# read from the FROZEN manifest, never recomputed: the ledger grows while the
# array runs, so a task that re-derived the pending list would resolve a
# different unit than the submitter counted.

set -uo pipefail

echo "---------- JOB INFOS ------------"
scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null || echo "(not in a SLURM job)"
pwd
nvidia-smi || true
echo -e "---------------------------------\n"

cd ..

PY=${PY:-../.venv/bin/python}
# Fail loudly and immediately if the interpreter is unusable. A worktree's own
# .venv often holds only pip+uv, and `PY=x` on its own line is a SHELL variable,
# not an environment one, so `--export=ALL,PY` would silently propagate nothing
# and every task would "succeed" having done no work.
if ! "$PY" -c "import jax, jaxley, optax" 2>/dev/null; then
    echo "FATAL: $PY cannot import jax/jaxley/optax." >&2
    echo "       export PY=/path/to/thesis/.venv/bin/python" >&2
    exit 1
fi

: "${STUDY_MANIFEST:?STUDY_MANIFEST is unset -- submit via ./submit_study.sh}"
CHUNK=${CHUNK:-1}
TASK=${SLURM_ARRAY_TASK_ID:-0}

SCRIPT=$("$PY" study.py --manifest "$STUDY_MANIFEST" --field script)
N_UNITS=$("$PY" study.py --manifest "$STUDY_MANIFEST" --count)

# Both queries MUST have produced something. An unreadable manifest otherwise
# leaves N_UNITS empty, which makes `((IDX >= N_UNITS))` true on the first
# iteration -- so the task breaks immediately and exits 0 having done no work.
# A whole array can "succeed" that way and produce nothing.
if [[ -z "$SCRIPT" || ! "$N_UNITS" =~ ^[0-9]+$ ]]; then
    echo "FATAL: cannot read the manifest at $STUDY_MANIFEST" >&2
    echo "       script='$SCRIPT' units='$N_UNITS' (cwd $(pwd))" >&2
    exit 1
fi
if ((N_UNITS == 0)); then
    echo "FATAL: manifest lists no units." >&2
    exit 1
fi

START=$((TASK * CHUNK))

echo "study manifest : $STUDY_MANIFEST"
echo "target script  : $SCRIPT"
echo "units          : $N_UNITS total, this task $START..$((START + CHUNK - 1))"

echo "-------- PYTHON OUTPUT ----------"
rc=0
for ((k = 0; k < CHUNK; k++)); do
    IDX=$((START + k))
    if ((IDX >= N_UNITS)); then break; fi

    CELL=$("$PY" study.py --manifest "$STUDY_MANIFEST" --index "$IDX" --field cell)
    # One argv token per line, read into an array. No eval and no quoting rules:
    # a value with a space would otherwise word-split into two flags, and the
    # study module rejects values containing a newline for exactly this reason.
    mapfile -t ARGS < <("$PY" study.py --manifest "$STUDY_MANIFEST" \
        --index "$IDX" --argv)
    if ((${#ARGS[@]} == 0)); then
        echo "!!! task $TASK unit $IDX: empty argv, skipping" >&2
        rc=1
        continue
    fi

    echo ""
    echo "=== task $TASK unit $IDX: $CELL ==="
    echo "    $SCRIPT ${ARGS[*]}"
    started=$SECONDS

    if "$PY" "$SCRIPT" "${ARGS[@]}"; then
        elapsed=$((SECONDS - started))
        # Record only on success, so a re-submission picks the failure back up.
        "$PY" study.py --manifest "$STUDY_MANIFEST" --index "$IDX" \
            --record --status ok --elapsed "$elapsed"
        echo "=== unit $IDX ok (${elapsed}s) ==="
    else
        unit_rc=$?
        echo "!!! unit $IDX FAILED (exit $unit_rc) -- continuing the chunk" >&2
        "$PY" study.py --manifest "$STUDY_MANIFEST" --index "$IDX" \
            --record --status failed || true
        rc=1
    fi
done
echo "---------------------------------"

# Finish the whole chunk even if one unit fails, but exit non-zero so sacct
# still flags the task.
exit $rc
