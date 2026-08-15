#!/bin/bash
#SBATCH --job-name=postA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --array=1-12
#SBATCH --output=job.%A_%a.postA.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Table A: post-level domain measures, one array task per month.
# Output: analysis_data/post_domain_measures_2020/month=NN.parquet
#
# Usage:
#   sbatch slurm/build_post_table.slurm
#   sbatch --array=3 slurm/build_post_table.slurm    # re-run a single month

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python -m gender_domain.build_post_table month --year=2020 --month=${SLURM_ARRAY_TASK_ID}
