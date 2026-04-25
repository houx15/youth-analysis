# Cross-lingual sentiment validation — China side

Reviewer concern: original weibo sentiment used Kimi + Chinese prompt; tweet sentiment used GPT + English prompt. Different LLMs and prompts could bias the comparison.

This runbook covers the **China side** only: sampling 200 weibos with their existing Kimi-Chinese opinion scores, then transferring the file to the US server. The downstream pipeline (DeepL translation → 5× GPT/Kimi re-scoring on the same translated text → mean diff + per-post LLM stability report) lives in the `twitterapi-io` repo and runs on the US server.

We sample once. The 5 rounds happen downstream on the same 200 posts with the same translations — that measures LLM stability, not sampling variance.

## What this side produces

`ai_attitudes/weibo_translation_sample.parquet` — 200 rows, columns `weibo_id, weibo_content, original_opinion, time_stamp`, proportional-stratified by `original_opinion` (-2/-1/0/1/2).

## Steps

### 1. Submit the SLURM job

From the project root on the China server:

```bash
sbatch slurm/sample_weibo_translation.slurm
```

The wrapper requests 1 hour, 16 GB, 4 cpus, with email-on-end. It runs `python sample_for_translation.py --sample_size 200`.

### 2. Verify the output

```bash
ls -lh ai_attitudes/weibo_translation_sample.parquet
tail slurm-<JOBID>.out
```

The log ends with a class distribution printout. Sanity check: total = 200, no class is wildly over- or under-represented.

### 3. Transfer to the US server

```bash
scp ai_attitudes/weibo_translation_sample.parquet \
    <user>@<us-server>:<path-to-twitterapi-io>/sentiment_results/cross_lingual/
```

Then follow `twitterapi-io/CROSS_LINGUAL_VALIDATION.md` on the US server.

## Files involved

- `sample_for_translation.py` — the sampling script (Fire CLI), `--sample_size`, `--seed`, `--output_file`
- `slurm/sample_weibo_translation.slurm` — SLURM submission wrapper
- `ai_sentiment_analyzer.py` — provides `get_group_date_range` and `BASE_DIR`
- Inputs read:
  - `ai_attitudes/ai_sentiment_results/analyze_all_results_group_*.csv`
  - `ai_attitudes/ai_weibo_text/<date>.parquet` for each date in the analyzed groups
- Output: `ai_attitudes/weibo_translation_sample.parquet`
