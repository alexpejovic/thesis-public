#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=20         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=1-00:00            # Runtime in D-HH:MM
#SBATCH --partition=cpu-galvani     # Partition to submit to
#SBATCH --mem=50G                 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/%j-allen_feats.out  # File to which STDOUT will be written
#SBATCH --error=logs/%j-allen_feats.err   # File to which STDERR will be written
#SBATCH --mail-type=FAIL           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=alex.pejovic@student.uni-tuebingen.de  # Email to which notifications will be sent

# print info about current job
echo "---------- JOB INFOS ------------"
scontrol show job $SLURM_JOB_ID
pwd
echo -e "---------------------------------\n"

cd ..


# Run code with values specified in task array
echo "-------- PYTHON OUTPUT ----------"
../.venv/bin/python allen_feats.py Pospischil
echo "---------------------------------"
