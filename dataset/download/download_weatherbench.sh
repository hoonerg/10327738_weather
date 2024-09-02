#!/bin/bash --login
#$ -cwd
#$ -t 1-93543
#$ -o /dev/null
#$ -e /dev/null

INDEX=$(expr $SGE_TASK_ID - 1)

module load apps/binapps/anaconda3/2023.03
source activate ldm

python download_weatherbench.py --task_id $INDEX