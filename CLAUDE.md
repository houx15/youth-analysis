# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current research scope (read first)

This repo has accumulated several historical analysis lines (youth users, AI attitudes, device switching, gender-norm word embeddings, newspaper corpora). **Only one topic is in scope right now: gender differences in visible information participation on 2020 Weibo** — public affairs vs. celebrity-centered popular culture, each split into *source retweeting* and *content expression*.

- Authoritative research design and statistical plan: `docs/research_design_code_audit_and_next_steps.md` (written by a colleague, in Chinese — leave it in Chinese). Theoretical framing: `docs/social_media_information_domain_participation_framework.md`. Read section 0 (the code-to-measure mapping table) before touching pipeline code.
- `docs/` is in `.gitignore`, so those files exist only on the local machine and never reach the server via git.
- **Out of scope** — do not modify or opportunistically refactor unless explicitly asked: `gender_norms/` (embeddings + newspaper gender-norm index), `ai_*.py` (AI attitudes/sentiment), `clean_youth_data.py`, `extract_youth_text.py`, `device_analysis.py`, `*.pyx` / `setup.py` (Cython user-ID extraction), `user_*_match*.py`.

## Execution environment: server + SLURM

The large datasets live only on the server (historical job path: `/gpfs/share/home/2401111059/youth_analysis/`). **`cleaned_weibo_cov/`, `merged_profiles/`, and `analysis_results/` do not exist locally, so anything that scans a full year of posts cannot run here.** The only locally runnable step is plotting from `viz_data/*.parquet`, which has already been downloaded.

- Submit compute via `sbatch`; never run heavy jobs on the login node (shared server).
- Put SLURM scripts in `slurm/` with a `.slurm` extension: `.gitignore` swallows `*.sh`, so only `slurm/*.slurm` gets tracked. Existing template: `slurm/sample_weibo_translation.slurm`.
- Each script starts with a purpose comment and a `sbatch slurm/<name>.slurm` usage line, then `source ~/miniconda3/etc/profile.d/conda.sh && conda activate opinion`. **All Weibo analysis on the server runs in the `opinion` conda env.**
- Existing job logs are named `job.<jobid>.<jobname>.out` (job names seen: `analysis`, `demographic`); logs are gitignored.
- Cost reference: one year of density computation took ~4h over 303 daily files single-process; the retweet analysis pulls many columns for a whole year into memory. Size jobs generously (≥64G) or shard by day/month.

## Common commands

Every analysis script uses a [fire](https://github.com/google/python-fire) CLI where the first positional argument is the subcommand.

```bash
# Data dictionary / sanity check before building anything (cheap, reads a few daily files)
sbatch slurm/inspect_cov_schema.slurm     # schema + column profile + downloadable sample
sbatch slurm/inspect_cov_users.slurm      # full-year unique users and gender composition
# -> analysis_data/schema_report/{schema,profile,users}_2020.json + sample_*.parquet

# 0) One-time: source account IDs (requires merged_profiles/merged_user_profiles.parquet)
python get_news_ids.py            # -> configs/news_user_ids.json (grouped into 8 source categories)
python get_entertain_ids.py       # -> configs/entertain_user_ids.json (grouped by 11 verification keywords)

# 1) Post-level base data (join profiles onto posts)
python weibo_cov_clean.py --year 2020            # youth_weibo_stat/*.parquet -> cleaned_weibo_cov/2020/*.parquet

# 2) Source retweet analysis (run once per domain)
python basic_text_analyzer.py year 2020 retweet --source_type news
python basic_text_analyzer.py year 2020 retweet --source_type entertain
python basic_text_analyzer.py province 2020 --source_type news      # province level
python basic_text_analyzer.py district 2020 --source_type news      # region level

# 3) Content density analysis
python news_density_analyzer.py analyze 2020
python entertain_density_analyzer.py analyze 2020
python entertain_name_analyzer.py analyze 2020        # per-celebrity mentions (extension)
python entertain_name_analyzer.py export_texts 2020

# 4) Export data for local plotting (run on server, then download viz_data/)
python export_viz_data.py all 2020

# 5) Local plotting (the only step that runs locally)
python visualize.py all 2020      # writes figures/{YYYYMMDD}_*.pdf
python visualize.py retweet 2020
python visualize.py density_user 2020
```

Recompute switches: `--force_reanalyze=True` for retweet analysis, `--force_recalculate=True` for density. Both skip work when a cache file exists.

Config: `cp configs/configs.example.py configs/configs.py`, then fill `OPENAI_API_KEY` / `OPENAI_BASE_URL` (`configs.py` is gitignored).

## Data flow and architecture

```
raw profile files -> basic_profile_extractor.py -> cleaned_profile_data/{date}.parquet
                  -> user_profile_analysis.py   -> merged_profiles/merged_user_profiles.parquet
                                                     |
raw post files -> basic_text_extractor.py -> youth_weibo_stat/{date}.parquet
                                                     |
                          weibo_cov_clean.py (left-join profiles)
                                                     v
                           cleaned_weibo_cov/2020/{date}.parquet   <- single input layer for all behavior analysis
                             /                       |                      \
    basic_text_analyzer.py          news_density_analyzer.py       entertain_density_analyzer.py
    (retweets of news/entertain)      (public-affairs vocab hits)    (celebrity vocab hits)
      analysis_results/retweet_*        analysis_results/post_density_*  analysis_results/entertain_post_density_*
                             \                       |                      /
                                     export_viz_data.py -> viz_data/*.parquet -> visualize.py -> figures/
```

The 2×2 measurement grid: domain (public affairs / celebrity culture) × participation mode (source retweeting / content expression).

- Source retweeting matches `r_user_id` against `configs/news_user_ids.json` (institutional media and government accounts; the 8 categories are defined in `configs/cn_news_sources_lists.py`) and `configs/entertain_user_ids.json` (identified from entertainment keywords in `verified_reason`).
- Content expression uses `configs/news_vocabulary_2020.txt` (816 terms) and `configs/entertainment_nouns_2020.txt` (535 celebrity names). `configs/` is the **single canonical location** for both — the legacy `wordlists/` path was removed from the loaders. **Both vocabularies and both account lists are already human-reviewed and finalized by the user — do not re-screen terms or regenerate candidates.** `build_entertain_vocab.py` and `check_entertainment_names.py` are the historical candidate-generation and LLM-assisted review steps, kept for provenance only.

### Key fields and conventions

- Key columns in `cleaned_weibo_cov`: `user_id` (int), `weibo_id`, `weibo_content`, `is_retweet`, `r_user_id`, `r_weibo_id`, `time_stamp`, `r_time_stamp`, `gender`, `province`, `region`, `demographic_*`.
- Type traps: `is_retweet` is the **string** `"1"`; `r_user_id` is a string while `user_id` is an int, so cast with `astype(str)` before comparing; account-ID JSON files also store strings.
- `gender` is `"m"` / `"f"` (some legacy helpers also accept `"男"/"女"`). Analysis sample: 225,339 users (173,503 female / 51,836 male).
- `province` arrives as a numeric code; each script carries its own copy of `PROVINCE_CODE_TO_NAME` and `DISTRICT_MAP` (East / Central / West / Northeast).
- Analysis and viz steps read/write parquet with `engine="fastparquet"`. **Always pass `columns=`** — never load a full table.
- Fixed plot colors: male `#20AEE6`, female `#ff7333`.

## Known measurement problems inherited from the current code

These come from section 0.6 of the research design doc. The new pipeline must fix them rather than copy them:

1. ~~**Vocabulary path mismatch**~~ — FIXED 2026-08-15: `entertain_density_analyzer.py` and `entertain_name_analyzer.py` now both read `configs/entertainment_nouns_{year}.txt` only, and `.gitignore` was given an exception so both vocabulary files are tracked. The reviewed file holds 535 entries, matching the count printed by the last server density job, so switching the path is not expected to change results — but the cached `entertain_post_density_2020.parquet` was produced from the old `wordlists/` copy and has not been byte-compared.
2. **Cache filenames carry only the year** (`analysis_results/post_density_2020.parquet`, etc.) with no vocabulary version or match-rule hash. Any vocabulary or rule change must invalidate the old cache.
3. **Denominators** — the existing PPT ratio is "domain retweeters ÷ all retweeters"; the paper also needs "÷ all same-gender users", with the denominator stated in column names and figure titles.
4. **Density post output is missing fields** — no date, no post type (original / retweet-with-comment / plain retweet), no matched terms, so a user–month panel cannot be built from it. The new post table must carry these.
5. **User-level density is a plain mean of post densities**; the paper also needs character-weighted density, topical-post share (the primary indicator), and hits per 1,000 characters — aggregation definitions must not vary silently across scripts.
6. **Matching rule** — density currently sums `str.count()` per term, so nested long/short terms double-count characters. Switch to longest-match-first with non-overlapping spans.
7. **Retweet delay** — currently record-level means with conventional standard errors. The paper needs outlier cleaning, user-level medians/quantiles, and user-clustered intervals.

## The `gender_domain/` package (new pipeline)

`gender_domain/` builds the four analysis tables the paper's models read. It is tested locally (`python3 -m pytest gender_domain/ -v`, 128 tests) because all measurement logic is pure; only the IO runs on the server.

| Module | Responsibility |
|---|---|
| `config.py` | Paths, vocabulary/account loading from `configs/`, SHA1 fingerprints, manifest writing |
| `text_rules.py` | Cleaning, post typing, the expressive-post definition, leftmost-longest matcher |
| `id_rules.py` | Canonical string user IDs (see the float trap below) |
| `build_post_table.py` | Table A, one row per post, month shards |
| `build_retweet_table.py` | Table B, one row per source retweet, month shards |
| `build_user_tables.py` | Tables C and D |
| `reconcile_baseline.py` | Compares the new tables against `viz_data/` |

Server run order — A and B are independent and can run together:

```bash
sbatch slurm/build_post_table.slurm       # array 1-12, table A
sbatch slurm/build_retweet_table.slurm    # array 1-12, table B
sbatch slurm/build_user_tables.slurm      # tables C and D, after both arrays finish
sbatch slurm/reconcile_baseline.slurm     # baseline comparison
# re-run one failed month: sbatch --array=7 slurm/build_post_table.slurm
```

Output: `analysis_data/{post_domain_measures,retweet_domain_events}_2020/month=NN.parquet` plus per-month manifests, then `user_domain_2020.parquet`, `user_month_domain_2020.parquet`, `reconciliation_2020.json`. Nothing here overwrites `analysis_results/` or `viz_data/` — those stay as the baseline.

### Definitions this pipeline fixes in place

- **Expressive post** = original post or retweet with an added comment, **and** at least one character after cleaning. Image-only and link-only posts are kept as rows but excluded from every content denominator (owner ruling, 2026-08-15). The manifest counts them so the excluded share is reportable by gender.
- **Plain retweets** (`转发微博`) count as diffusion — they are in `n_retweets` and in source events — but never as expression.
- **Retweet chain** = everything from `//@` to end of post, stripped before URLs so a URL abutting the chain cannot shield it.
- **Matching** is leftmost-longest and non-overlapping, so nested vocabulary terms never double-count characters.
- **Zero denominators yield NaN, never 0** — "no expressive posts" and "0% topical" are different facts.
- `topical_share` uses expressive posts; `topical_share_allposts` keeps the literal all-posts denominator for comparison with the old pipeline.

### Operational notes

- **Table A shards from before 2026-08-15 are unusable.** They lack the `is_expressive` column, and `build_user_tables` will fail hard with `pyarrow.lib.ArrowInvalid: No match for FieldRef.Name(is_expressive)`. That is intended — rebuild the shards rather than working around it. There is no warning message to look for.
- **User IDs**: if a daily file has any null `user_id`, pandas upcasts the column to float and a naive `astype(str)` yields `"123.0"`, silently breaking every account-list match and cross-table join. Always go through `id_rules.normalize_id_series`. It refuses values above 2^53, where a float round-trip would return a *wrong* ID rather than a suffixed one.
- **Manifest counter naming differs between A and B**: Table A separates `rows_dropped_within_day_dedup` from `rows_dropped_cross_file_dedup`; Table B reports both under the latter. The identities still reconcile within each table.
- Both tables dedup within a month shard only, so a `weibo_id` spanning a month boundary survives twice.
- Every run records git SHA, a dirty-tree flag, input shards, vocabulary/account fingerprints, and per-stage row counts. Tables C and D inherit fingerprints from the shard manifests rather than re-reading `configs/`, so they describe the vocabulary that actually produced the numbers.

Every production run should record: input files, code version, vocabulary version, account-list version, parameters, and step-by-step sample counts.

## The results layer

On top of the four tables, `gender_domain/` also computes the paper's results and figures (design doc §6–12). 438 tests, all runnable locally.

| Module | Produces |
|---|---|
| `profile_join.py` | Province + M2 profile controls on table C, with per-gender sample-loss accounting |
| `stats_utils.py` | Wilson/Newcombe intervals, cluster bootstrap, average marginal effects, the shared result schema |
| `describe.py` | 表 1 sample activity, 表 2 raw gender gaps |
| `models_core.py` | §6.2–6.5 entry, intensity, share, persistence across M0/M1/M2 |
| `models_interaction.py` | §6.6 `Gender × Domain` difference-in-differences — the paper's central claim |
| `models_combination.py` | §7 source/content decomposition, §8 participation combinations |
| `models_temporal.py` | §9 monthly persistence and leave-one-month-out, §10 delay quantiles |
| `export_figure_data.py` / `figures.py` | Small downloadable export, then figures drawn locally |

```bash
sbatch slurm/run_results.slurm          # profile join -> describe -> models -> temporal
sbatch slurm/export_figure_data.slurm   # only after run_results succeeds
# download analysis_data/figure_data/ and analysis_data/results/, then locally:
python3 -m gender_domain.figures all --year=2020
```

### Reporting rules this layer enforces

- **Marginal effects on the probability/proportion scale are the reported quantity**, never odds ratios alone — ORs are not comparable across the nested M0/M1/M2 layers the paper shows side by side.
- **Every estimate carries a 95% CI.** At n=225,339 significance is uninformative, so effect sizes lead. A missing interval is publishable; an under-covering one is not — helpers return NaN with a note rather than a narrow interval they cannot justify.
- **NaN is never zero.** A user with no expressive posts is excluded from that outcome with a recorded reason, not counted as 0%.
- **Every ratio names its denominator**, in a column or the `model` label.
- **Failures leave a trace**: a model that does not converge emits a NaN row with a note, never a missing row — a missing row reads as a hypothesis nobody tested.
- SEs are robust for user-level models and clustered by user wherever a user contributes multiple rows.
- **No single news-vs-entertainment preference score** (§8.3) — enforced by tests that scan outputs and module source.

### Operational notes

- **`run_results.slurm` stamps a run id into every result file and the export verifies they match.** A stale table from an earlier run is refused by name rather than silently copied into the figures. Both SLURM scripts use `set -euo pipefail`; do not remove it — without it a crashed stage still reports success and later stages publish the previous run's numbers.
- `(outcome, domain, model, term)` is unique across every result file; variants live inside `model` (`M1/share_ge_0.1`, delay-threshold variants). Use `figures.select_one()`, which raises on a multi-row match.
- `domain="both"` marks the difference-in-differences row. A `groupby`/`dropna` on domain would silently swallow the headline.
- `models_temporal` aligns cluster groups via `model.data.row_labels`; passing the group column name fails with `cov_type="cluster"` in statsmodels 0.14.2 (verified — the column-name form is GEE-only).
- Measured: results layer runs in well under an hour; the delay stage dominates. `models_core` at full scale is 7–10 min, 1.6 GB.
- Known open decisions, none affecting a published number: `models_intensity` re-embeds entry rows (self-contained, but double-counts if the four core tables are concatenated); `ts_date_mismatch_share` is computed post-cleaning while described as an independent health check; delay rows switch from record to user units.

## Code style

`AGENTS.md` holds the existing conventions (fire CLI pattern, import grouping, naming, memory/chunking rules) and still applies — consult it before changing code instead of duplicating it here. Note that existing code comments, docstrings, and `print` logging are in Chinese; keep new code consistent with the surrounding file. Documentation written for the user is in English; `docs/` (authored by a colleague) stays in Chinese.
