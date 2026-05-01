#!/bin/sh 
### General options 
#BSUB -q hpc
#BSUB -J jobMaritimeShippingAISDKRelbenchTasks
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=27GB]"
##BSUB -R "select[model == XeonGold6226R]"
#BSUB -M 27GB
#BSUB -W 2:00 
#BSUB -u s242947@dtu.dk
#BSUB -B 
#BSUB -N 
#BSUB -o jobMaritimeShippingAISDKRelbenchTasks_%J.out 
#BSUB -e jobMaritimeShippingAISDKRelbenchTasks_%J.err 

# Load Python and CUDA
module load python3/3.11.13 
module load cuda/11.7

# Create virtual environment (only if missing)
if [ ! -d /dtu/blackhole/08/213928/venv ]; then
    python3 -m venv /dtu/blackhole/08/213928/venv
fi

# Activate your virtual environment 
source /dtu/blackhole/08/213928/venv/bin/activate 

# Move to your project folder 
cd /dtu/blackhole/08/213928/storage

# Run your script

python maritime_shipping_ais_relbench_tasks.py
