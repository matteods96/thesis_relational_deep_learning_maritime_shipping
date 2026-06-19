#!/bin/sh 
### General options 
#BSUB -q hpc
#BSUB -J jobMaritimeShippingAISDKRelbenchDatasetNew
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=180GB]"
#BSUB -M 180GB
#BSUB -W 2:00 
#BSUB -u s242947@dtu.dk
#BSUB -B 
#BSUB -N 
#BSUB -o jobMaritimeShippingAISDKRelbenchDatasetNew_%J.out 
#BSUB -e jobMaritimeShippingAISDKRelbenchDatasetNew_%J.err 

# Load Python and CUDA
module load python3/3.11.13 
module load cuda/11.7

# Create virtual environment (only if missing)
if [ ! -d /dtu/blackhole/08/213928/venv ]; then
    python3 -m venv /dtu/blackhole/08/213928/venv
fi

# Activate your virtual environment 
source /dtu/blackhole/08/213928/venv/bin/activate 

# Install required Python packages INSIDE the venv
pip install --upgrade pip
pip install duckdb requests geopandas pyarrow openpyxl relbench

# Move to your project folder 
cd /dtu/blackhole/08/213928/storage

# Run your script
python maritime_shipping_ais_relbench_dataset_new.py