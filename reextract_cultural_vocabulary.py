"""复用已完成的全年热搜缓存重新提词；修改下面两个配置即可换批次。"""

import json
import os
from pathlib import Path

from cultural_participation.reextract import refresh


ROOT = Path(__file__).resolve().parent
SOURCE_DB = ROOT / "gender_norms/newspaper_data/cultural_participation/pilot_2020_17028/topics.sqlite"
RUN_LABEL = "full_2020_v2"


if __name__ == "__main__":
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ValueError("请通过 sbatch reextract_cultural_vocabulary.sh 提交")
    output = ROOT / "gender_norms/newspaper_data/cultural_participation" / f"{RUN_LABEL}_{job_id}"
    print(f"原缓存：{SOURCE_DB}\n新版输出：{output}", flush=True)
    print(json.dumps(refresh(SOURCE_DB, output), ensure_ascii=False), flush=True)
