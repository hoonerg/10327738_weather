#!/bin/bash --login
#$ -cwd
#$ -l nvidia_v100=2
#$ -pe smp.pe 8

module load apps/binapps/anaconda3/2023.03
source activate ldm

python generate_samples.py