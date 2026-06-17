#!/bin/sh 
#BSUB -q hpc
#BSUB -J jobVisualInspectionShiptypesFirst180Days
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=27GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -M 27GB
#BSUB -W 24:00 
#BSUB -u s242947@dtu.dk
#BSUB -B 
#BSUB -N 

#BSUB -o jobVisualInspectionShiptypesFirst180Days_%J.out
#BSUB -e jobVisualInspectionShiptypesFirst180Days_%J.err

nvidia-smi

module load python3/3.11.13
module load cuda/11.7

source /dtu/blackhole/08/213928/venv/bin/activate

cd /dtu/blackhole/08/213928/storage

python visual_inspection_shiptypes_first_180_days_2025.py