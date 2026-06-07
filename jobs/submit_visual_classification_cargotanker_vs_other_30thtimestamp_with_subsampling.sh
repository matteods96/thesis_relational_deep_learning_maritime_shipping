#!/bin/sh
#BSUB -q hpc
#BSUB -J jobVisualClassificationCargotankerVsOther30thTimestampWithSubSampling
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=27GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -M 27GB
#BSUB -W 24:00
#BSUB -u s242947@dtu.dk
#BSUB -B
#BSUB -N

#BSUB -o jobVisualClassificationCargotankerVsOther30thTimestampWithSubSampling_%J.out
#BSUB -e jobVisualClassificationCargotankerVsOther30thTimestampWithSubSampling_%J.err

nvidia-smi

module load python3/3.11.13
module load cuda/11.7

source /dtu/blackhole/08/213928/venv/bin/activate

cd /dtu/blackhole/08/213928/storage
python visual_classification_cargotanker_vs_other_30thtimestamp_with_subsampling.py


python 