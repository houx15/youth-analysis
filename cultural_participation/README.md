# 性别化文化参与：跨领域分析

本模块采用“热搜辅助选词，在用户原始文本中测量领域参与”的路线。热搜不是最终参与分析的样本边界。先留在现有仓库，通过独立包与配置隔离新流程；没有旧分析模块依赖，未来可单独迁移。

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

模型输出仅作为审核材料。分类任务将词条和例句明确当作数据，不执行其中的指令。实际分类由独立的 `classify` 命令调用 OpenRouter；结果通过结构校验后保存为待审核材料，尚无正式词表导出功能。

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
sbatch prepare_cultural_vocabulary.sh /path/to/bangdan_data/2020 run_2020_v1
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

同一份任务文件可以传给不同模型，避免词表、例句和说明变化干扰比较。`classify` 每次只接收一个 `--model`，无默认模型。运行环境通过 `OPENROUTER_API_KEY` 提供密钥，不写入源码或结果。

接口采用 [OpenRouter Chat Completions](https://openrouter.ai/docs/api_reference/overview)，要求所选模型端点支持 [JSON Schema 结构化输出](https://openrouter.ai/docs/guides/features/structured-outputs)，并设置 `require_parameters=true`。模型返回后另行校验词条遗漏、重复、额外词、领域标签以及“都不是”互斥规则。

```bash
# 在已配置密钥的环境执行；MODEL_ID 替换为明确选择的 OpenRouter 模型 ID。
# 初次只跑一批，检查输出与 usage；完整任务仍须通过 SLURM。
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

行为识别默认区分大小写、字面子串匹配，保留所有嵌套命中；每帖每领域只计一次，多领域可以同时命中。它不计算不重叠字数密度。词表子抽样可以精确保留短词命中，但仍需要抽查误匹配；上下文条件在删词时保持不变。

### 北大服务器运行

从仓库根目录 `git pull --ff-only origin main`。脚本默认激活 `~/miniconda3` 下的 `opinion` 环境，路径／环境名可用 `CULTURAL_CONDA_INIT`、`CULTURAL_CONDA_ENV` 覆盖。不预设分区或账号，可在 sbatch 参数里指定。环境需要 `jieba`（提词）、`numpy`（统计与语义）、`pyarrow`（parquet）。

词表准备：

```bash
# 参数1为已解压的榜单目录；参数3可选，为全局最多读取行数。
sbatch prepare_cultural_vocabulary.sh /path/to/bangdan_data/2020 pilot_2020 1000
sbatch prepare_cultural_vocabulary.sh /path/to/bangdan_data/2020 full_2020
```

后续内容行为分析（以下路径是占位示例；run 输出目录必须尚不存在）：

```bash
sbatch run_cultural_analysis.sh build \
  --input-dir cleaned_weibo_cov/2020 \
  --vocabulary /path/to/reviewed_vocabulary.json \
  --output gender_norms/newspaper_data/cultural_participation/behavior_v1 --year 2020

# build 完成后单独提交，不能与前一步无依赖并发运行。
sbatch run_cultural_analysis.sh summarize \
  --db gender_norms/newspaper_data/cultural_participation/behavior_v1/behavior.sqlite \
  --output gender_norms/newspaper_data/cultural_participation/summary_v1

sbatch run_cultural_analysis.sh subsample \
  --db gender_norms/newspaper_data/cultural_participation/behavior_v1/behavior.sqlite \
  --output gender_norms/newspaper_data/cultural_participation/subsample_v1 \
  --fractions 0.5 0.8 0.9 --repeats 100 --seed 2020
```

输入从 parquet 每次只读八个必要字段、默认5000行，不能用 pandas 整年载入。SQLite 在磁盘保存全局去重帖子及命中关系，汇总使用磁盘临时表。全量运行需要足够磁盘空间；当前单作业无断点续跑，失败输出不具备 complete 标记，不能继续汇总。不要用月度数组后直接加总用户比例；不同月份用户重叠，必须统一分母。

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
