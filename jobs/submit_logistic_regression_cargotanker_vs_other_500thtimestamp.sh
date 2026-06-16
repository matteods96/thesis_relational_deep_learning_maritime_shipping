
#!/bin/sh
#BSUB -q gpua100
#BSUB -J jobLogisticRegressionCargoTankerVSOther_500thTimeStamp
#BSUB -n 4
##BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -gpu "num=1"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=128GB]"
#BSUB -M 128GB
#BSUB -W 24:00
#BSUB -u s242947@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o LogisticRegressionCargoTankerVSOther_500thTimeStamp_%J.out
#BSUB -e LogisticRegressionCargoTankerVSOther_500thTimeStamp_%J.err

nvidia-smi

module load python3/3.11.13
module load cuda/11.7

source /dtu/blackhole/08/213928/venv/bin/activate

cd /dtu/blackhole/08/213928/storage

python logistic_regression_cargotanker_vs_other_500thtimestamp.py