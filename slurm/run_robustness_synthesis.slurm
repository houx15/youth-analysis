#!/bin/bash
#SBATCH --job-name=robustsyn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=job.%j.robustsyn.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=2401111059@stu.pku.edu.cn

# §13 稳健性套件的综合层：读五个族的结果表 -> 四条准则 + FDR + 规格曲线。
# 产出 analysis_data/robustness/synthesis*.parquet（九张表）。
#
# 用法：
#   sbatch slurm/run_robustness_synthesis.slurm
#   # 串在两个数组作业后面（run_all.slurm 就是这么投的）：
#   sbatch --dependency=afterok:<重量数组 id>:<轻量数组 id> --export=ALL \
#          slurm/run_robustness_synthesis.slurm
#
# 为什么单独一个脚本、而不是塞进任一个数组：
# 综合层必须等五个族**全部**写完才有意义，而数组任务之间没有顺序保证。
# 原来的做法是让人在登录节点手跑（它确实只要几秒到几分钟），但那一步没有
# 任何东西保证五个族真的都跑完了——手跑早了，load_all 会按"缺哪族就少哪族"
# 的口径出一份看起来正常、其实少了两族的综合表。做成作业之后，afterok
# 依赖替人守住这个前提。
#
# 额度：本身只读五张 parquet，几秒钟就能算完；给到 4 小时是因为
# load_all(allow_recompute=True) 在基线行缺失时会从表 C 现算一次基线，
# 那一步要把用户级母表读进内存。内存峰值几十 G——本集群不接受 --mem，
# 内存跟着分区走，这里记的是实测占用而非申请额度。
#
# 运行标识：与五个族共用同一个 GENDER_DOMAIN_RUN_ID（由调用方 export，
# run_all.slurm 会传下来）。缺失时退回本作业 id——那会让综合层的 manifest
# 与五个族的对不上，事后无法证明它们属于同一批，所以尽量用 --export=ALL。

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opinion

YEAR="${YEAR:-2020}"
export GENDER_DOMAIN_RUN_ID="${GENDER_DOMAIN_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d%H%M%S)}}"
echo "GENDER_DOMAIN_RUN_ID=${GENDER_DOMAIN_RUN_ID}"

python -m gender_domain.robustness.synthesis build --year="${YEAR}"

echo "综合层完成，run_id=${GENDER_DOMAIN_RUN_ID}"
