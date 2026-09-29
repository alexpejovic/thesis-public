#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=20         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-01:00            # Runtime in D-HH:MM
#SBATCH --partition=a100-galvani     # Partition to submit to
#SBATCH --gres=gpu:1              # optionally type and number of gpus
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/%j-region_crossval.out  # File to which STDOUT will be written
#SBATCH --error=logs/%j-region_crossval.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=alex.pejovic@student.uni-tuebingen.de  # Email to which notifications will be sent

# Cross-validate gen_region_success_scores against a row already on disk, then
# calibrate adam's learning rate. Both are GPU-bound and short.
#
#   cd encoding/slurm_scripts && sbatch ./region_crossval.sh

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi
echo -e "---------------------------------\n"

cd ..

echo "-------- CROSS-VALIDATION -------"
../.venv/bin/python tests/test_crossval_betweenset.py "$@"
CROSSVAL_RC=$?
echo "crossval rc=$CROSSVAL_RC"
echo "---------------------------------"

if [ "${SKIP_ADAM_SWEEP:-0}" = "1" ]; then
    echo "-------- ADAM LR SWEEP SKIPPED --"
    exit $CROSSVAL_RC
fi

echo "-------- ADAM LR SWEEP ----------"
# Adam's per-coordinate step is ~lr, so polyak's max_learning_rate=1.0 would
# move ~50 units over 100 cosine-decayed steps inside a +-1.5 box. Pick the lr
# minimising d_final.p50 subject to moving_frac < 0.25 and escape_frac < 0.05.
for LR in 0.02 0.05 0.1 0.2; do
    ../.venv/bin/python ./scripts/datagen/gen_region_success_scores.py \
        --set_key gLeak+eLeak+gNa --loss encoder-zscore-mask \
        --optimizer adam --max_learning_rate "$LR" \
        --ideal_seeds 0 3 --init_seeds 0 64 \
        --out ./data/tuning/adam-lr-sweep.jsonl --no-save_raw
done
echo "---------------------------------"

exit $CROSSVAL_RC
