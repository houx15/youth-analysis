# 性别化文化参与：跨领域分析

本模块采用“热搜辅助选词，在用户原始文本中测量领域参与”的路线。热搜不是最终参与分析的样本边界。先留在现有仓库，通过独立包与配置隔离新流程；没有旧分析模块依赖，未来可单独迁移。

## 已保存的阶段词表

2026-10-13形成严格版词表`strict-v1-2026-10-13`：3,972个独立词，覆盖五个领域，仅纳入本轮裁决可独立使用的1,001词，以及双模型领域组合一致且双方均判standalone的2,971词。需要语境和模糊词按已有分类标记暂不纳入。

- [阶段报告：筛选流程、领域规模、代表词与高频词](reports/2026-10-13-vocabulary-v1.md)
- [词表 JSON](vocabularies/strict_v1/vocabulary.json) / [CSV](vocabularies/strict_v1/vocabulary.csv)
- [统计与输入哈希](vocabularies/strict_v1/summary.json)

词表快照作为版本化研究配置入Git，原始响应和审核工作数据仍不入库。人物关联参与与话题内容参与需区别解释；本版尚未在原微博中验证准确率。复算入口为`python -m cultural_participation.finalize_vocabulary`，参数见阶段报告。

## 本轮已实现

1. 逐行读取已解压的 `weibo_bangdan.*` 文件，兼容旧脚本的第二列 JSON 和嵌套 bangdan 格式，过滤广告。
2. 磁盘 SQLite 保存完整标题、来源文件、行号、抓取时间和榜单类型。相同标题全局合并，但每次出现的来源仍保留；标题相同不代表事件相同。
3. 从完整标题提取名词、动词候选，保留词性和标题关联。不限制中文或最长四字，不强制 top-5000；此阶段不进行语义筛除。默认最短两字符。
4. **先导出候选词 CSV 和规模报告**：总候选词数、导出词数、长度分布、标题频次分布、不同频次门槛下剩余词数、词条及例句字符数和文件字节数。
5. **独立的可选步骤**：仅在检查规模、完成必要规则过滤后，读取已确认的 CSV，按词条生成离线模型任务。每词附最多三条标题例句及独立标题数；任务记录词表与分类体系哈希。此步骤不在默认批处理流程内。

预算不能只根据词数计算；需要结合模型、提示词、例句、预计输出及重试开销。当前报告只提供可核实的规模，不把字符数当作 token 或虚报费用。规则过滤可先使用 `export --min-titles N` 另存新版，数据库保留全部候选；不同门槛的剩余词数只是规模诊断，不自动决定筛词阈值。

`distinct_title_count` 是包含某词的不同标题数量，不是上榜次数、用户数或微博词频。重复抓取不会提高它。默认只读取 `type=1` 实时榜，缺失类型会计入排除行；可显式指定 `--board-type all`。解析失败和广告等计数随运行返回并写入数据库。正式运行前应在真实小样本上确认格式、缺失及过滤比例。

候选提词仍受 jieba 分词边界影响，可能漏掉长短语；目前是第一版候选生成基线。例句按标题排序取前三条，不能视为语境的代表性样本，歧义词后续需补充语境。

## 分类定义

`taxonomy.json` 是可修改草案：政治、经济商业、科学技术、体育、文艺娱乐。

- `domains=["none"]`：**都不是**，不属于任何目标领域；不可与目标领域并选。
- `domains=[]`：**无法判断**，信息不足或歧义未消除，在理由里解释。
- 允许多个目标领域并选。
- 单独判断适用性：`standalone` 可独立匹配；`context_required` 需要语境；`exclude` 为通用或无效词。
- “都不是”和“排除”不是同一维度。例如健康词可能含义明确但不在当前目标体系，应可建议新增健康领域。

模型输出仅作为审核材料。分类任务将词条和例句明确当作数据，不执行其中的指令。实际分类由独立的 `classify` 命令调用 OpenRouter；结果通过结构校验后保存为待审核材料。完成裁决后，可用独立的`finalize_vocabulary`入口按明确纳入规则合并导出，分类命令本身不自动批准词条。

## 小样本使用

在仓库根目录执行。仅提词阶段依赖 jieba；其余准备步骤使用 Python 标准库。可在分析环境中安装 `jieba==0.42.1`，运行会记录实际版本。此包不读取旧配置文件或 API 密钥。

```bash
python -m cultural_participation collect \
  --input-dir /path/to/bangdan_data/2020 \
  --db gender_norms/newspaper_data/cultural_participation/pilot.sqlite \
  --max-lines 1000

python -m cultural_participation extract \
  --db gender_norms/newspaper_data/cultural_participation/pilot.sqlite

python -m cultural_participation export \
  --db gender_norms/newspaper_data/cultural_participation/pilot.sqlite \
  --output gender_norms/newspaper_data/cultural_participation/pilot_candidates.csv
```

输入目录是已有解压目录，本模块不自动解压。每次收集和任务导出使用新路径，已有文件会报错，避免意外覆盖。提词可在同一数据库重跑，会在事务内替换候选关联。

全量收集、提词和分析必须通过根目录的 `prepare_cultural_vocabulary.sh` 提交 SLURM。数据输出沿用项目约定，进入被忽略的 `gender_norms/newspaper_data/cultural_participation/`，不入 Git。规模核对后才考虑下一阶段。

```bash
sbatch prepare_cultural_vocabulary.sh
python -m unittest discover -s cultural_participation/tests -v
```

## 大模型准备是单独一步

先检查 `pilot_candidates.csv` 和旁边的 `.summary.json`，决定是否过滤。以下命令只读取明确指定的候选词表，不会绕过过滤重新使用数据库全部候选。

```bash
python -m cultural_participation prepare \
  --candidates gender_norms/newspaper_data/cultural_participation/pilot_candidates.csv \
  --output gender_norms/newspaper_data/cultural_participation/pilot_jobs.jsonl
```

修改分类草案后应重新生成任务，使用新文件名。真正调用模型是下面的独立阶段，尚未进行真实调用。

## OpenRouter 单模型运行与一致性比较

2026-10-10 运行约定：联网分类直接运行，不使用 SLURM；默认每次一个词。
两个模型选择 `deepseek/deepseek-v4.1-flash` 和 `qwen/qwen3.8-flash`，
请求及运行记录明确设置 `reasoning.enabled=false`。
旧的多词任务文件必须重新 prepare，默认值变化不会修改已有任务。
两个模型共用同一个新任务文件；36,876 个词各需 36,876 次请求。
先各运行 `--max-batches 1` 检查接口，再决定全量。
原始语料处理、提词及其他资源密集分析仍使用 SLURM。

### tmux 直接运行（每次一个词）

从服务器仓库根目录执行 `git pull --ff-only origin main` 和 `tmux new -s cultural`。
以下步骤均在新 tmux 会话中执行。`RUN_DIR` 替换为新版 candidates.csv 的实际目录。

```bash
module load anaconda/3.11
source ~/.bash_profile
conda activate opinion
python -m pip install tqdm

read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
export OPENROUTER_API_KEY
echo

RUN_DIR="gender_norms/newspaper_data/cultural_participation/full_2020_v2_替换作业ID"
JOBS="$RUN_DIR/classification_single_jobs.jsonl"
OUTPUT="gender_norms/newspaper_data/cultural_participation/model_runs"

python -m cultural_participation prepare \
  --candidates "$RUN_DIR/candidates.csv" --output "$JOBS" --batch-size 1
```

prepare 只执行一次；已有同名任务不会被覆盖，两个模型复用此文件。
先各测试一个词（产生费用）：

```bash
for MODEL in deepseek/deepseek-v4.1-flash qwen/qwen3.8-flash; do
  python -u -m cultural_participation classify \
    --jobs "$JOBS" --output "$OUTPUT" --model "$MODEL" --max-batches 1
done
```

确认两个模型结果均为 `ok: 1`，再顺序执行全量：

```bash
TOTAL=$(wc -l < "$JOBS")
for MODEL in deepseek/deepseek-v4.1-flash qwen/qwen3.8-flash; do
  python -u -m cultural_participation classify \
    --jobs "$JOBS" --output "$OUTPUT" --model "$MODEL" --max-batches "$TOTAL"
done
```

tqdm 显示百分比、完成批数、速率、预计剩余时间和 ok/error 计数；单词任务一批就是一个词。
每次响应落盘后更新进度；等待 API 返回期间进度数字不变，单请求超时为 180 秒。
测试运行与全量运行分别保存；全量会重新请求测试过的第一个词。
按 Ctrl-b 然后按 d 脱离会话；重新连接用 `tmux attach -t cultural`。
tmux 断开不影响运行，但进程被终止后当前代码不能断点续跑；不要重复启动全量来恢复。

同一份任务文件可以传给不同模型，避免词表、例句和说明变化干扰比较。`classify` 每次只接收一个 `--model`，无默认模型。运行环境通过 `OPENROUTER_API_KEY` 提供密钥，不写入源码或结果。

接口采用 [OpenRouter Chat Completions](https://openrouter.ai/docs/api_reference/overview)，要求所选模型端点支持 [JSON Schema 结构化输出](https://openrouter.ai/docs/guides/features/structured-outputs)，并设置 `require_parameters=true`。模型返回后另行校验词条遗漏、重复、额外词、领域标签以及“都不是”互斥规则。

```bash
# 在已配置密钥的环境执行；MODEL_ID 替换为明确选择的 OpenRouter 模型 ID。
# 初次只跑一个词，检查输出与 usage；联网分类可直接运行。
python -m cultural_participation classify \
  --jobs /path/to/classification_jobs.jsonl \
  --output gender_norms/newspaper_data/cultural_participation/model_runs \
  --model MODEL_ID --max-batches 1
```

结果路径为 `输出根目录/URL编码后的完整模型ID/UTC日期/时分秒_运行ID/`。例如模型 ID 内的 `/` 编码为 `%2F`，避免不同命名映射到同一目录。每次生成独立目录，不覆盖同日旧运行。

- `run.json`：请求模型 ID、UTC 时间、任务路径、批数与输出 token 上限。
- `responses.jsonl`：逐批原始响应（包括服务返回的实际模型与 usage）、输入任务、请求参数、校验结果和错误。不保存认证头。
- 不自动重试；失败保留记录，网络中断时不能推断是否计费。尚未实现断点续跑。请求次数上限由 `--max-batches` 明确指定，不等同于金额预算。

```bash
python -m cultural_participation compare \
  --responses /path/to/model_A/run/responses.jsonl /path/to/model_B/run/responses.jsonl \
  --output gender_norms/newspaper_data/cultural_participation/comparison_v1
```

比较通过任务哈希及词条对齐，只对完全相同任务中双方有效的条目计算领域标签集合的精确一致率、适用性一致率；同时报告有效与未配对条目数、每次运行成功／失败批数、“都不是”分歧和“无法判断”分歧。输出 `summary.json` 与 `disagreements.jsonl`。没有共同有效条目时一致率为 null，不能记为零或一。

该指标是模型间的一致率，不是准确率或因果证据。不同批次划分、例句、提示词或任务文件内容将导致任务哈希不同，不作直接配对；两个模型应复用完全相同的任务文件。旧版要求返回数组的任务文件应重新 prepare，以采用当前 results 对象格式。

## 后续边界

人工审核与最终词表由研究者完成，接口见下文。原始文本匹配、参与汇总、跨领域比较、词表子抽样和语义评分已实现第一版，见“后续分析代码”。模型供应商为 OpenRouter，具体模型、最终领域数和中文语义轴仍由研究者确定。

验证目前为人工构造数据和模拟 API 响应上的过滤、来源追踪、重复计数、小样本限额、任务输出、文件保护、模型路径隔离、响应校验和一致率测试；尚未进行真实榜单全量运行、真实 API 调用或评估分类准确率。

## 后续分析代码（2026-10-08）

已实现可运行的第一版：`vocabulary.py` 正式词表与匹配，`behavior.py` 内容行为命中缓存，`analysis.py` 用户汇总、跨领域比较与词表子抽样，`semantics.py` 语义轴／调查对照，统一入口为 `python -m cultural_participation.research`。代码尚未在北大服务器全年语料上验证性能；当前模型是可检查的描述性基线，不是旧版全部统计模型的逐项移植。

### 给词表处理的交接接口

正式输入是 JSON，见 `examples/vocabulary.example.json`（**仅展示格式，不是研究用词表**）。顶层含 `version`、`domains` 和 `terms`。

每词一行对象：`term`、`domains`（多标签列表）、`decision`、`approved: true`；可加 `kind`（如 person/topic/organization），需要上下文的词必须配置 `context_any`。上下文条件为清理后同一正文中至少出现一个指定字符串，不是分类模型的自由文本理由。`none`、尚未审核和 exclude 词条不进入正式词表。领域体系可以包含细分领域，ID 不必限于最初五个草案标签。

行为识别默认区分大小写、字面匹配，保留中文嵌套命中；新增ASCII字母、数字及下划线边界检查，避免AI命中RAISE、NBA命中WNBA内部，中文相邻不阻断匹配。每帖每领域只计一次，多领域可以同时命中。它不计算不重叠字数密度。词表子抽样可以精确保留短词命中，但仍需要抽查误匹配；上下文条件在删词时保持不变。

### 北大服务器运行

从仓库根目录 `git pull --ff-only origin main`。两个 SLURM 脚本均按用户指定顺序初始化：`module load anaconda/3.11`、`source ~/.bash_profile`、`conda activate opinion`，然后调用 Python。词表路径与规模由 Python 读取固定配置。两个新作业脚本沿用用户已验证的北大模板：`-p C032M0128G`、`--qos=low`、`--nodes=1`、`--ntasks-per-node=8`，不设置 `--mem` 或 `--cpus-per-task`。当前 Python 流程仍是单进程，不因申请了8个任务而自动启动8份分析。环境需要 `jieba`（提词）、`numpy`（统计与语义）、`pyarrow`（parquet）。

词表准备：

```bash
# 配置已写入 cultural_participation/vocabulary_job.conf：
# INPUT_DIR="bangdan_data/2020"，RUN_LABEL="pilot_2020"，MAX_LINES=1000。
sbatch prepare_cultural_vocabulary.sh
# 改全量时，在配置中将 MAX_LINES 改为0、RUN_LABEL改为full_2020，然后仍用同一命令。
```

### strict_v1在北大服务器启动（分月Parquet版）

2026-10-15改用旧分析的效率设计：按月独立处理、内存批量匹配、顺序写Parquet、年度分片汇总。取消旧任务后从原始文件重新开始；不读取旧SQLite缓存。五领域词表及行为指标保持不变。原始输入是`cleaned_weibo_cov/2020/YYYY-MM-DD.parquet`，月份按文件名调度，每条记录仍按北京时间戳过滤年份。

在仓库根目录运行：

```bash
git pull --ff-only origin main
module load anaconda/3.11
source ~/.bash_profile
conda activate opinion
python -c "import numpy, pandas, pyarrow, ahocorasick; print('依赖齐全')"
# 如仅缺ahocorasick：python -m pip install pyahocorasick
bash submit_cultural_behavior.sh
```

配置在`cultural_participation/behavior_job.conf`，默认全量2020，无截取行数限制。输入路径不同只改INPUT_DIR；全新运行使用新的RUN_LABEL。提交脚本只调用sbatch，所有数据处理都在SLURM计算节点运行；保持用户验证过的分区、qos、ntasks-per-node=8和conda初始化顺序。每个作业运行一个Python进程。

提交脚本同时识别Slurm的`Submitted batch job 123`和Crane的`Job id allocated: 123.`，不依赖`--parsable`。如果只有月度任务提交成功，用`bash submit_cultural_behavior.sh --month-job 123`接续；如果用户汇总已提交，用`--reduce-job 456`只补交最后一步。不要重新提交已经在运行的阶段。

运行期间若仅需修复提交脚本，可`git fetch origin main`后用`git show origin/main:submit_cultural_behavior.sh > submit_cultural_behavior.sh`单独更新该文件。不要在月份运行期间切换HEAD或更新分析代码，以免月份manifest的代码版本不一致。

三个阶段自动通过afterok依赖衔接：

1. **12个月份任务，默认最多3个并行**。每批读取8个必要字段，默认50,000行。C实现的Aho-Corasick自动机保留大小写、ASCII边界和重叠词命中；有界缓存复用重复原文的匹配结果。写出每帖特征、命中词列表及用户分片，不逐条写数据库。
2. **64个年度用户分片，默认最多4个并行**。同一用户全年落在同一分片，在内存中去重、检查性别冲突、汇总年度分子分母，再计算比例。只读取该分片的精简特征，不载入全年原始文本。另用轻量帖子ID分片检查同一帖子是否错误地归属多个用户。BUCKET_COUNT越大，单个汇总任务内存一般越少；分片数必须在首次运行前确定。
3. **统计汇总任务**。分批读取用户表，输出描述统计、性别差异和跨领域差异；通过两遍批量充分统计量计算OLS与HC1标准误。每个用户等权。

BATCH_SIZE控制原始文本处理的批大小，不控制模型或抽样；MONTH_CONCURRENCY/REDUCE_CONCURRENCY控制同时运行的任务数。日志使用普通文件开始/完成和阶段进度，不使用tqdm。示例：

```bash
# 将12345换成提交脚本显示的月度任务ID，1代表一月份。
tail -f job.12345.1.cultural_month.out
squeue -u "$USER"
```

不要在运行中修改代码、词表或配置。完成的月份/用户分片有manifest及输入版本信息；同配置再次提交会跳过这些分片。失败的临时目录不会作为成功结果读取。如任务失败，先取消旧依赖链中仍等待的汇总任务，再修正问题并重新提交；代码或输入发生变化时改RUN_LABEL从头运行，避免混合版本。月份没有文件会记录为空月；全年没有匹配文件则报错。若某月份失败，可以单独`sbatch --array=月份 run_cultural_behavior.sh`，之后再提交完整依赖链。磁盘需容纳特征和命中词Parquet；旧SQLite文件由用户自行清理。

结果默认位于：

```text
gender_norms/newspaper_data/cultural_participation/behavior_parquet_v2_2020/
  months/month=01/manifest.json
  months/month=01/vocabulary.json
  months/month=01/posts-000.parquet     # 精简帖子特征和命中词
  months/month=01/owners-000.parquet    # 全局帖子归属检查
  users/bucket=000/users.parquet
  users/bucket=000/manifest.json
  summary/user_domain.csv
  summary/descriptives.csv
  summary/gender_gaps.csv
  summary/cross_domain_contrasts.csv
  summary/coverage.csv
  summary/user_exclusions.csv
  summary/manifest.json
```

需要字段：weibo_id、user_id、gender、is_retweet、weibo_content、r_weibo_content、time_stamp、r_time_stamp。字段缺失报错；年份外、缺ID、无效转发标记的行数记入月度manifest；去重和性别排除记入用户分片manifest。coverage中的total_posts、total_retweets、total_expression分别对应旧输出posts、retweets、expressive_posts。汇总目录已存在则拒绝重复提交，防止覆盖结果。

合成数据测试比较新旧用户指标、性别差异、跨领域差异和HC1标准误，并检查跨月重复、性别冲突及帖子归属冲突。尚未在北大全年数据上验证速度和峰值内存，不承诺具体完成时间。原SQLite版`research build/summarize/subsample`保留供旧结果兼容，不用于这次全量运行。新Parquet保存逐词命中，但**删词重采样入口尚未迁移**，不能将新分片传给旧的`subsample --db`。

### 行为口径与输出

- 研究总体：指定年份内实际观测到至少一帖且用户性别为一致 m/f 的账户。性别缺失、冲突账户单列报告；不自动推定机构账号身份，也不声称代表所有微博用户。
- 去重：全局 `weibo_id`，按排序后的输入文件保留第一条。跨用户 ID 冲突直接报错；同用户重复版本采用首条，记重复数。
- 转发：`is_retweet` 判定，领域由 `r_weibo_content` 清理后的内容判定；该列整体缺失则报错，不回退为来源账号分类。原文缺失的转发仍进入全部转发分母，另报可取得原文的数量。
- 表达：`weibo_content` 删除转发链和链接后的本人文字；纯转发占位文字及空文本不进入有效表达分母。
- 转发时间：秒／毫秒归一化，使用北京时间筛选年份；非正延迟及缺失值保留状态、不进入时滞均值。输出 `log_delay` 为用户在该领域正时滞转发的平均 log(1+秒)，是发生转发条件下的指标，不是曝光反应速度。
- 进入概率：该领域是否至少一次参与，分母为全部有效用户；份额分别以该用户全部转发、全部有效表达为分母。零分母记缺失，不记零。
- `comment_on_retweet_share`：该领域转发中，用户新增文字也命中同领域的比例；泛泛的“哈哈”不能据原帖算为同领域表达。
- 性别差异为女性减男性，用户等权。不是直接对比男女人数或女性占比。

`summary_v1` 包含 `user_domain.csv`、`descriptives.csv`、`gender_gaps.csv`、`cross_domain_contrasts.csv`、`coverage.csv`、`user_exclusions.csv` 和口径／来源 manifest。

统计基线为逐领域 OLS（进入变量对应线性概率模型），同时输出原始与控制 log(1+发帖数)、log(1+转发数)、log(1+活跃天数) 后的差异，使用 HC1 标准误。奇异矩阵或样本不足明确留空并标记。暂未接入人口画像控制、原版 logit AME、机构排除或因果模型。区间是点态区间，不作大量跨领域显著性声明。

跨领域对比以同一用户的“领域A指标减领域B指标”为结果，再计算性别差异，保留用户内部的相关性。份额比较使用共同分母；时滞对比只包含在两个领域均有有效时滞的用户，样本可能不同，结果单列人数。

子抽样按全词表无放回抽取，固定随机种子，保存每次保留的词与领域性别描述统计；`repeat=-1` 是完整词表基准。小领域可能被抽空；此时明确得到零命中，不悄悄删掉领域。子抽样结果衡量词表构成敏感性，不是总体抽样置信区间。

### 语义轴与调查验证接口

```bash
sbatch run_cultural_analysis.sh score \
  --vectors /path/to/chinese_word_vectors.vec \
  --axes /path/to/validated_axes.json \
  --objects /path/to/cultural_objects.json \
  --output gender_norms/newspaper_data/cultural_participation/semantic_v1

sbatch run_cultural_analysis.sh survey-check \
  --scores /path/to/semantic_v1/scores.csv \
  --ratings /path/to/survey_ratings.csv \
  --output gender_norms/newspaper_data/cultural_participation/survey_check_v1
```

`axes.json` 格式为 `{"gender":{"positive":["锚词A"],"negative":["锚词B"]},"prestige":{...}}`；由研究者确定正负方向及经验证的中文锚词。此处不会内置未经验证的中文“声望词轴”，不会把 cultivation、potency、morality 自动混合为 prestige。

`objects.json` 是 `[{"object_id":"唯一ID","domain":"领域ID","terms":["对象词1","对象词2"]}]`。行为识别词不自动成为语义锚词或文化对象，二者需要明确对应。向量仅支持有／无首行维度声明的 word2vec 文本格式，逐行扫描、只保留所需词，记录向量全文哈希。锚词缺失直接报错；对象词缺失显示覆盖率，全部缺失留空，不插补。对象分数为各对象词与归一化语义轴的余弦均值。

调查输入 CSV 为 `object_id,axis,rating`，每行一条有效评分（不了解留空）；编码方向应与 embedding 一致。代码输出逐对象平均评分及对象层面的 Pearson 相关，仅作无权重描述性对照，未实现复杂抽样权重、评价者分组或时间差校正。文化女性化与低声望的静态关联不自动等于已识别动态贬值过程。

本轮本地验证：人工语料、真实小型 parquet 分批读取、性别冲突、缺失分母、转发链、嵌套词重采样、秒／毫秒时滞、HC1与直接矩阵计算对照、模拟向量与调查连接。未运行真实全年数据、未调用付费模型、未选择最终中文语义轴。


同层级的行为—语义连接：

```bash
sbatch run_cultural_analysis.sh link-behavior \
  --gaps /path/to/summary_v1/gender_gaps.csv \
  --scores /path/to/semantic_v1/scores.csv \
  --output gender_norms/newspaper_data/cultural_participation/map_v1 \
  --metric expression_share --model unadjusted_OLS
```

只将行为词表的领域 ID 与语义对象 `object_id` 完全一致的项配对，输出行为差异／性别关联／声望得分的 `map.csv` 及描述相关；不将“体育”的参与差异复制给“足球”等多个子类。如果研究对象下沉到子类，行为词表也应先在相同层级上编码。正负方向由轴配置定义，一定先核对性别轴的正端是否为女性。小样本领域相关及低向量覆盖不能直接作为贬值结论。

## 提词 v2：2026-10-10 全年词表检查后的修正

用户确认第一版35,291词覆盖全年日期。v2保留原名词／动词候选、低频词和二字词，补充：

- 英文缩写与数字字母组合，如 NBA、CBA、GDP、5G、iPhone12；NBA7人中的NBA也保留，纯数字和URL不作为此规则候选。组合和字母部分均为待审核线索，不宣称自动完成实体识别。
- 书名号内的2–40字符短语。
- `extraction_rules.json` 中从实际标题确认的少量完整词形（优衣库、华鼎奖、天问一号），仅当原标题实际包含它们时补入。可扩充该规则表；不自动拼接任意相邻词。仍不能保证完整修复中文分词。

导出新增 `extraction_sources`、`review_flags`、`containing_phrases_json`。标记低频、纯动词、可能片段及已观察完整短语的组成部分，**只供审核，不自动删词**。例如保留“优衣”并提示其来源标题还出现“优衣库”。现有LLM prepare可直接读新版CSV，并把审核标记与完整词形提示传给模型；不同模型继续使用同一个prepare文件。

每个词每个独立标题仍只计一次，同词被多条规则提取不会重复增加频次。新版规模报告包含各提取来源与审核标记的词数，类别可重叠。

已完成全年收集时，直接使用新版重提词脚本：

```bash
git pull --ff-only origin main
sbatch reextract_cultural_vocabulary.sh
```

脚本沿用已跑通的调度和Anaconda初始化方式。`reextract_cultural_vocabulary.py` 已将输入设为 `gender_norms/newspaper_data/cultural_participation/pilot_2020_17028/topics.sqlite`，输出为 `full_2020_v2_<作业ID>/`。如果旧数据库被移动，修改该Python文件顶部的 `SOURCE_DB`。

此步骤只复制独立标题和来源元信息到新库、重新提词，不重扫305个原始文件，也不覆盖旧库。抓取明细仍保存在旧库，新manifest记录原库位置；新库的occurrences表不复制原抓取明细。新结果仍需对比分词覆盖和规模，不预先保证词数或精度。`prepare_cultural_vocabulary.sh` 全流程也已使用新版提词逻辑。

### 2026-10-16 行为结果与论文图

已基于服务器汇总包完成五领域的描述、活动量调整前后对比、转发时间与同域附言、跨领域份额差异分析。详见[阶段报告](reports/20261016_behavior_analysis.md)和[四页PDF图集](reports/20261016_cultural_participation_figures.pdf)。图沿用既有性别配色、标记、字体与日期前缀；所有图中的估计和区间来自服务器结果，没有在本地重新拟合模型。

复现：`python3 -m cultural_participation.result_report --date 20261016`。输入是`outputs/summary_review.tar.gz`；小型结果表、PDF和报告入库，原始大数据不入库。阶段报告包含针对本轮数据的解释，生成器用输入哈希防止新数据误用旧叙述。


### 2026-10-16 全词表语义测量与词项行为联结

首轮使用SGNS Mixed-large及fastText CC+Wiki的显式词向量，全部3,972词进入审计，缺失不填零。中文候选轴见`semantic_specs/axes_zh_v1.json`；报告见[语义阶段报告](reports/20261016_semantic_analysis.md)及[六页PDF](reports/20261016_semantic_figures.pdf)。人物保留，排除检验只针对已有明确标记，不代表完成全词表实体分类。

词项行为分析复用原月度Parquet中的`retweet_terms`和`expression_terms`，无需重扫原始文本或使用SQLite。先确认`behavior_job.conf`的`OUTPUT_ROOT`与`RUN_LABEL`指向已有完整月度及用户分桶结果。当前提交脚本对应64个用户桶，同时运行4个任务。不要在运行过程中更新代码或修改词表。

在服务器仓库根目录执行：

```bash
git pull --ff-only origin main
sbatch run_cultural_term_behavior.sh
```

等64个数组任务全部成功后，再手动提交：

```bash
sbatch summarize_cultural_terms.sh
```

汇总程序会检查所有分桶、来源及词表一致性。它输出全部词项的描述统计，以及转发参与、表达参与、份额、转发时间和附言指标的未调整／活动量调整性别差异。无可估计结果保留状态，不以零替代。程序逐桶读入，避免全年用户×词项矩阵。原始年度行为结果不覆盖。

完成后下载日志末尾显示的`term_summary_review.tar.gz`，放到本地`cultural_participation/outputs/`。其中包含`term_descriptives.csv`、`term_gender_gaps.csv`、`term_coverage.csv`和`manifest.json`。这一步尚未在服务器执行；本地合成数据已检验与既有单词行为分析结果一致。只有拿到这些词项结果后，才能联结行为与语义，不能把五领域系数复制给各词。

词项汇总包返回后，将四个文件解压到独立目录（例如`outputs/term_summary_v1/`），运行：

```bash
python -m cultural_participation.behavior_semantic_link \
  --behavior cultural_participation/outputs/term_summary_v1 \
  --date 20261016
```

日期改为实际分析日期。该入口核对行为／语义词表哈希、词项集合及指标完整性，生成词项联结审计表、总体与领域内Pearson/Spearman描述、稀疏度／已标记人物排除检验，以及8页PDF。20和100名参与用户只是稀疏度敏感性阈值，主分析保留全部有向量且可估计词项。它不把跨词系数当作独立观测做常规显著性检验。当前只完成程序验证，尚无真实词项行为联结结果。

### 2026-10-16：互斥主词表与等规模稳健性词表

两版本均从strict_v1删除全部134个跨领域词，不重新分配领域。

|版本|用途|政治|商业|科技|体育|文艺|总词数|
|---|---|---:|---:|---:|---:|---:|---:|
|exclusive_full_v1|主分析|746|581|251|700|1560|3838|
|exclusive_top250_v1|词表规模稳健性|250|250|250|250|250|1250|

输入分别为`vocabularies/exclusive_full_v1/vocabulary.json`及`vocabularies/exclusive_top250_v1/vocabulary.json`。后者从前者按`distinct_title_count`降序取各领域前250，并列按词项Unicode顺序。频率是包含该词的不同热搜标题数，不是原微博词频。各目录包含CSV及记录来源哈希、筛选规则和排除词的summary.json。

生成器：`python -m cultural_participation.derive_vocabularies`，已有目录时拒绝覆盖。已核对接口兼容、互斥、数量、子集关系和排序。原词表保留，新词表行为结果尚未重算；当前运行配置不修改，避免影响已有任务。两版本必须分开存储结果，不能混用旧词表哈希或旧领域统计。复用逐帖命中词重聚合时，同帖同域多个词仍只计一次。

等规模版本用于检验对词表规模及高频词选择的敏感性，不能预先称已确认词数不影响结果。词项互斥不要求微博内容只涉及单一领域。

### 两版原始微博重跑：月度频次、分布、活动量匹配、年龄／地区

本轮按用户要求从`cleaned_weibo_cov/2020`重新扫描，两套词表独立匹配，不读取旧运行的逐帖命中结果。配置为`cultural_participation/extended_job.conf`，新输出根目录为`gender_norms/newspaper_data/cultural_participation/exclusive_behavior_v1_2020/`。原始月度处理沿用已验证的批读取和64用户桶架构，不使用数据库，不使用tqdm。3个月为一个提交任务，每月顺序处理两版词表。

先核对配置中的原始数据、画像CSV和ID映射路径。画像默认沿用旧`user_profile_analysis.py`的`/gpfs/share/home/2401111059/cov-weibo-2/COV-Weibo2.0/user.csv`及`data/weibo_cov_user_to_original_id.json`。读取CSV的user_id、birthday、province三列，分块过滤到本轮用户；先用映射恢复原始微博ID。如果使用已是原始微博ID的CSV，将ID_MAPPING设为空字符串。不会根据年龄预先删除主样本。

在仓库根目录依次执行：

```bash
git pull --ff-only origin main
sbatch run_cultural_exclusive_q1.sh
sbatch run_cultural_exclusive_q2.sh
sbatch run_cultural_exclusive_q3.sh
sbatch run_cultural_exclusive_q4.sh
```

等四个任务全部成功（两版各12个月）后：

```bash
sbatch reduce_cultural_exclusive.sh
```

等64个数组分片全部成功后：

```bash
sbatch summarize_cultural_exclusive.sh
```

不使用自动依赖或`--parsable`。整个运行期间不修改代码、配置和词表，避免月份来源版本不一致。失败任务可原命令重提，完成且版本一致的月份、分桶、匹配与汇总可跳过；不删除已成功结果。请勿将新任务指向旧输出目录。配置BUCKET_COUNT固定64，与分桶提交脚本一致；改动时必须同步调整数组范围。

**输出口径：**

- `monthly_users/`：每个年度有效用户补齐12个月。每月领域表达、转发次数；不参与记零；无分母的份额和无有效间隔的时间指标记缺失。跨月重复帖先全年去重，再按真实发帖月份归属。
- `extended_summary/descriptives.csv`：各月及年度用户统计的均值、标准差、标准误、零值比例、有效用户数。`annual_user`下表达／转发次数为每用户全年次数除12；时间指标为用户全年有效转发间隔均值，不按12除。秒数与log(1+秒)均输出；秒数由逐帖缓存log间隔还原。
- `extended_summary/distributions.csv`：固定区间频数、比例、累计比例，覆盖零值和长尾，区间为左闭右开。不把直方图近似当作精确中位数／分位数。
- `extended_summary/gender_gaps.csv`：相应分组女性均值减男性均值；当前匹配与分层比较报告描述性均值差，不输出未经配对处理的差异置信区间。描述表中的标准误是单个性别的均值标准误，不是配对差异的标准误。
- `annual_summary/`：继续输出原来的年度进入、份额、时间、同域附言，以及未调整／活动量调整OLS和跨领域比较。该结果覆盖全体样本，不是匹配样本回归。

**匹配：**使用全年总发帖、总转发、活跃天数的log1p值，以0.25为固定宽度分箱；在三维分箱完全一致的格内按固定随机种子20261016男女1:1无放回抽取。它是粗化精确匹配，不是倾向得分匹配；领域行为不进入匹配变量。两版年度用户及活动量须完全相同，否则中止；两版共用同一匹配名单。`sample/matching_balance.csv`提供原始尺度与log尺度的均值及SMD，SMD使用匹配前合并组内标准差作为固定分母。必须检查实际保留人数及平衡，不能因运行成功就宣称匹配充分。`stratified_balance.csv`另列年龄／地区内的log活动量平衡；全局匹配不保证各分层内部也平衡。

**画像：**年龄=2020减出生年，分为0–13、14–15、16–17、18–20、21–30、31+，同时输出单岁年龄分布。无效生日、未知地区单列Unknown；冲突生日或冲突大区设缺失。地区采用旧项目东部／中部／西部／东北四大区，接受省份名称和两位代码。无画像者仍进入总体结果。`profile_coverage.csv`与`sample_coverage.csv`报告分性别有效率、匹配保留及分层人数。画像无任何ID交集时直接报错，提醒修正路径／ID映射。

最终下载新根目录下`exclusive_review.tar.gz`，其中有两版小型年度和扩展结果，以及匹配、画像诊断；不含用户明细和原始数据。本地还没有本轮真实结果，服务器运行后再作比较和日期前缀PDF绘图。不要把前一轮多领域词表的结果称作本轮结果。

服务器Crane包装器拒绝在sbatch脚本名后附多个参数，因此月份固定在q1–q4四个提交脚本内。请先进入仓库根目录，再无参数提交；q1为1–3月、q2为4–6月、q3为7–9月、q4为10–12月。输出路径与原设计完全相同。

### 2026-10-16：两版结果已返回并核验

`outputs/exclusive_review.tar.gz`已在本地分析。详见[两版行为阶段报告](reports/20261016_exclusive_analysis.md)与[八页PDF](reports/20261016_exclusive_figures.pdf)。复现：`python -m cultural_participation.exclusive_report`，输出日期前缀图表、来源审计及小型结果表；分布CSV分块读取。生成器锁定本包哈希，防止新数据沿用旧解释。

匹配49,837对用户；整体活动量均值平衡良好。两版匹配样本中五领域表达／转发频次差异方向全部保留，已知四大区也重复同样方向。词表缩减仍影响绝对水平及差距，时间间隔的算术均值和平均log结果存在差别。年龄字段算出的0–9岁占76.02%，尚待确认生日含义，不能直接当真实年龄解释；14–15、16–17记录组内部也存在匹配后的活动量残差。原始年龄分层表保留作诊断。已知日期缺口不阻止分析，月度图将无输入的10月显示为空档。
