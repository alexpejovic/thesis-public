# Studies: spec-driven SLURM orchestration

A **study** is a cartesian product of parameter axes run against one existing
entry point. Adding one means writing a YAML file in
`encoding/configs/studies/`, not writing a new SLURM script.

```bash
cd encoding/slurm_scripts

./submit_study.sh --list                     # what studies exist
./submit_study.sh loss_sweeps --dry-run       # preflight + plan, no submission
./submit_study.sh loss_sweeps --pilot         # a couple of units, to measure
./submit_study.sh loss_sweeps                 # the whole study
./submit_study.sh loss_sweeps                 # again later: resumes what is left
```

The pieces:

| File | Role |
| --- | --- |
| `encoding/configs/studies/<name>.yaml` | the spec: axes, fixed flags, resources |
| `encoding/study.py` | expands the spec, freezes manifests, owns the ledger |
| `slurm_scripts/submit_study.sh` | preflight, work plan, QOS guards, submission |
| `slurm_scripts/study_array.sh` | the one array body every study shares |

## Why not just loop `sbatch`

`mass_gen_losses.sh` and its siblings hardcode a value list in bash and
re-submit every value on every invocation. That is fine for 27 one-off jobs and
poor for anything larger. This adds three things:

1. **A frozen manifest per submission.** `--materialize` writes the pending unit
   list to `data/studies/<name>/<datestr>/jobs.jsonl`, and array tasks index into
   *that*. Recomputing the pending list per task would be a real bug: the ledger
   grows while the array runs, so task 7 would resolve a different unit than the
   submitter counted, and units would be skipped or run twice.
2. **Resume by ledger.** Each finished unit appends a row to
   `data/study-runs.jsonl` keyed by the spec's `key` fields. Re-running the same
   command submits only what is missing — the target script needs no resume
   logic of its own. Only successes are recorded as `ok`, so a failed unit comes
   back on the next submission.
3. **Preflight before submission.** Every target builds its flags from its own
   YAML config, so a misspelled axis name is an argparse error *in every task at
   once*. `--validate` catches it, along with the two traps below, before
   anything is queued.

## Two traps the validator exists to catch

**Unquoted datestrs.** YAML 1.1 reads `2026_08_12_15_20_52_343885` as an integer
with underscore digit separators, silently yielding `20260812152052343885`. Quote
every datestr. The validator compares each supplied value's type against the
base config's default and rejects the mismatch.

**List-valued config keys.** `_parse_args` uses `type=type(v)`, so a key whose
YAML default is a list becomes `type=list` and argparse shreds `-3 3` into
single characters with no error. `opt_bounds` and `freeze_params` are both like
this: change them in the base YAML, never through a study axis.

## Spec reference

```yaml
description: One line, shown by --list.
script: ./scripts/datagen/generate_losses.py   # relative to encoding/
base_config: ./configs/loss_sweep.yaml         # validated against

axes:                        # ordered; the product is the unit list
  encoder_datestr: ["2026_08_12_12_30_21_338929"]   # quote datestrs
  seed: {range: [1001, 1005]}                       # half-open
  # third form: {from_file: ./configs/studies/dates.txt}  (one per line)

flags:                       # fixed on every unit
  channel_type: Pospischil

key: [encoder_datestr, seed] # ledger identity; renaming orphans old rows
ledger: ./data/study-runs.jsonl
inputs: [./data/voltage_traces/Pospischil.dat]      # preflight existence checks

resources:
  partition: a100-galvani
  gres: gpu:1
  cpus_per_task: 20
  mem: 50G
  time_per_unit: 0-00:30     # per unit; the submitter multiplies by the chunk

chunk: null                  # units per task; null = derive from MaxSubmit
cap: 8                       # max simultaneously running tasks
```

Resources become `sbatch` CLI flags, which override the `#SBATCH` header in
`study_array.sh`. That header is only a fallback for a manual run.

## QOS guards

Verified with `sacctmgr show qos format=Name,MaxSubmitJobsPerUser,MaxJobsPerUser,MaxWall`
— `normal`/`gpunormal` allow **MaxSubmit 300**, **MaxJobs 36**, **MaxWall 72 h**.
The submitter refuses to exceed the first two and warns on the third; override
with `MAX_SUBMIT`, `MAX_RUNNING`, `MAX_WALL_H`. MaxSubmit is checked against the
*live* `squeue` count, not against an empty queue.

`chunk` exists because of MaxSubmit: a 5,000-unit study cannot be 5,000 array
tasks, so each task runs `chunk` units in sequence and `--time` is multiplied to
match.

## Monitoring

```bash
squeue -j <jobid>
sacct -j <jobid> --state=FAILED,TIMEOUT --format=JobID,State,Elapsed
less logs/<jobid>_<task>-<study>.out
./submit_study.sh <study> --dry-run        # progress: re-reads the ledger
```

A unit that fails does not abort its chunk; the task finishes the rest and still
exits non-zero so `sacct` flags it.

## `PY`

`submit_study.sh` resolves the interpreter to an absolute path and exports it,
because `--export=ALL,PY` only propagates *environment* variables — a bare
assignment would silently reach the tasks unset, and a worktree's own `.venv`
often holds only pip and uv. It resolves with `realpath -ms`: `.venv/bin/python`
is a symlink to the base conda interpreter, and following it breaks
`site-packages`. Override with `PY=/path/to/.venv/bin/python`.

## Adding a study

1. Copy a spec in `configs/studies/`, point `script` and `base_config` at your
   target, and list the axes.
2. `./submit_study.sh <name> --dry-run` until the preflight is clean.
3. `./submit_study.sh <name> --pilot`, then read the measured elapsed times out
   of the ledger and correct `time_per_unit`.
4. `./submit_study.sh <name>`.

No Python changes are needed for a new study. `study.py` knows nothing about
losses, encoders or convexity.
