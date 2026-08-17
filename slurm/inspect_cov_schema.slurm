#!/bin/bash
#SBATCH --job-name=schema
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=job.%j.schema.out
#SBATCH --mail-type=end
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# Build a data dictionary for cleaned_weibo_cov/{year} before the main pipeline:
# schema consistency across daily files, per-column missingness, categorical value
# counts, unique-key check, timestamp unit, post-type composition, and a small
# downloadable sample. Reads only a handful of daily files - no heavy compute.
# Output: analysis_data/schema_report/{schema,profile}_{year}.json + sample_*.parquet
#
# Usage:
#   sbatch slurm/inspect_cov_schema.slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

python inspect_cov_schema.py all 2020 --n_files=3
