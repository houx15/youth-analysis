#!/bin/bash
#SBATCH -o job.%j.cultural_exclusive_q4.out
#SBATCH -p C032M0128G
#SBATCH --qos=low
#SBATCH -J exclusive_q4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
module load anaconda/3.11
source ~/.bash_profile
conda activate opinion || exit 1
set -e
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source cultural_participation/extended_job.conf
test -d "$INPUT_DIR" || { echo "原始数据目录不存在：$INPUT_DIR"; exit 1; }
test -f "$PROFILE_CSV" || { echo "画像CSV不存在：$PROFILE_CSV"; exit 1; }
if [ -n "$ID_MAPPING" ]; then test -f "$ID_MAPPING" || { echo "ID映射不存在：$ID_MAPPING"; exit 1; }; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# 月份固定在脚本内，提交时无需参数。
for month in 10 11 12; do
    for version in exclusive_full_v1 exclusive_top250_v1; do
        python -u -m cultural_participation.parquet_behavior month \
          --input-dir "$INPUT_DIR" --year "$YEAR" --month "$month" \
          --vocabulary "cultural_participation/vocabularies/$version/vocabulary.json" \
          --output "$OUTPUT_ROOT/$RUN_LABEL/$version/months" \
          --bucket-count "$BUCKET_COUNT" --batch-size "$BATCH_SIZE"
    done
done
