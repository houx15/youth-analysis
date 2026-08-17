#!/bin/bash
#SBATCH --job-name=rtB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --array=1-12
#SBATCH --output=job.%A_%a.rtB.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Table B: source retweet events, one array task per month.
# Output: analysis_data/retweet_domain_events_2020/month=NN.parquet
#
# Usage:
#   sbatch slurm/build_retweet_table.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python -m gender_domain.build_retweet_table month --year=2020 --month=${SLURM_ARRAY_TASK_ID}
