#!/bin/bash --login
#$ -cwd

module load apps/binapps/anaconda3/2023.03
source activate ldm

python process.py