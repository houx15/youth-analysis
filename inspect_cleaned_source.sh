#!/bin/bash
#SBATCH -o job.%j.source_inspection.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J source_inspect
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion || exit 1
cd /gpfs/share/home/2401111059/youth_analysis || exit 1
python -u -m cultural_participation.inspect_cleaned_source
