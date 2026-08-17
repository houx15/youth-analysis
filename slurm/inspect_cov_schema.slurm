#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32

# Build a data dictionary for cleaned_weibo_cov/{year} before the main pipeline:
# schema consistency across daily files, per-column missingness, categorical value
# counts, unique-key check, timestamp unit, post-type composition, and a small
# downloadable sample. Reads only a handful of daily files - no heavy compute.
# Output: analysis_data/schema_report/{schema,profile}_{year}.json + sample_*.parquet
#
# Usage:
#   sbatch slurm/inspect_cov_schema.slurm

module load anaconda/3.11
source ~/.bash_profile
conda activate opinion

python inspect_cov_schema.py all 2020 --n_files=3
