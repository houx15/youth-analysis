#!/bin/bash
#SBATCH -o job.%j.cultural_reextract.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J cultural_reextract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion
python reextract_cultural_vocabulary.py
