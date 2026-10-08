#!/bin/bash
#SBATCH -o job.%j.cultural_vocab.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J cultural_vocab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion
python prepare_cultural_vocabulary.py
