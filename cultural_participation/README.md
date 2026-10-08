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

后续模块再接入人工审核和版本化词表导出、原始文本匹配、参与汇总及词表子抽样敏感性分析。正式测量需独立定义转发原文、用户新增文字和时间分母。领域数与模型供应商尚未定稿，本轮不提前固定。

验证目前为人工构造数据和模拟 API 响应上的过滤、来源追踪、重复计数、小样本限额、任务输出、文件保护、模型路径隔离、响应校验和一致率测试；尚未进行真实榜单全量运行、真实 API 调用或评估分类准确率。
