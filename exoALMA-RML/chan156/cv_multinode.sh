#!/bin/bash
#SBATCH --job-name=crossvalidate
#SBATCH --time=1:00:00
#SBATCH --partition=YOUR_RESOURCE_HERE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --mem=6G
#SBATCH --chdir=/rml/LkCa15/LkCa15_12CO_100ms_LSRK_contsub
#SBATCH --output=/rml/LkCa15/LkCa15_12CO_100ms_LSRK_contsub/slurm_output/slurm-%j.out
#SBATCH --array=0-104%15

# load appropriate modules
module purge
module load gcc/12.2.0
module load ffmpeg/5.1.2
module load openmpi/4.1.5
source YOUR_ENVIRONMENT

date
echo "Job id: $SLURM_JOBID"
echo "Changing into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR
python cv_single_tune.py $SLURM_ARRAY_TASK_ID
echo "Complete"
date