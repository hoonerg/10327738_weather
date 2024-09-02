#!/bin/bash --login
#$ -cwd
#$ -l nvidia_a100=1
#$ -pe smp.pe 12

module load apps/binapps/anaconda3/2023.03
source activate ldm

python train_processor.py