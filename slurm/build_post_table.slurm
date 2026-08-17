#!/bin/bash
#SBATCH -o job.%A_%a.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --array=1-12

# Table A: post-level domain measures, one array task per month.
# Output: analysis_data/post_domain_measures_2020/month=NN.parquet
#
# Usage:
#   sbatch slurm/build_post_table.slurm
#   sbatch --array=3 slurm/build_post_table.slurm    # re-run a single month

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion

python -m gender_domain.build_post_table month --year=2020 --month=${SLURM_ARRAY_TASK_ID}
