#!/bin/bash
#SBATCH -o job.%A_%a.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --array=1-12

# Table B: source retweet events, one array task per month.
# Output: analysis_data/retweet_domain_events_2020/month=NN.parquet
#
# Usage:
#   sbatch slurm/build_retweet_table.slurm

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion

python -m gender_domain.build_retweet_table month --year=2020 --month=${SLURM_ARRAY_TASK_ID}
