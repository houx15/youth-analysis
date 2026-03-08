# 报纸数据省级Word2Vec分析计划

## 项目概述

基于报纸数据复现省级Word2Vec嵌入分析，参考Weibo数据的处理流程。

### 数据源
- 路径：`/lustre/home/2401111059/newspaper_data/pdf_txt`
- 格式：JSONL文件（每行一个JSON对象）
- 规模：211个文件，约数十万到数百万篇报纸文章
- 关键字段：
  - `source`: 报纸名称（如"经济日报"）
  - `pdf_txt`: 文章正文
  - `title`: 标题
  - `pdf_url`: 原文链接

### 输出目标
- 各省份的Word2Vec模型（与Weibo分析相同结构）
- 省级性别规范指数（gender norm index）

---

## 技术挑战与解决方案

### 挑战1：计算节点无网络访问
**问题**：计算节点无法访问Wikipedia或其他在线资源来获取报纸的省份信息。

**解决方案**：
1. 在计算节点提取所有报纸名称列表（无网络需求）
2. 在有网络的环境（登录节点或本地）使用Wikipedia API获取报纸-省份映射
3. 保存映射为JSON文件
4. 在计算节点使用映射文件进行数据处理

### 挑战2：大规模数据处理
**问题**：报纸数据量可能非常大（单文件70万+文章），需要内存优化。

**解决方案**：
- 使用逐行处理，不将整个文件加载到内存
- 使用生成器和迭代器模式
- 流式处理和写入

---

## 执行步骤

### Step 0: 探索数据（完成）
- [x] 了解数据格式
- [x] 统计文件数量和规模
- [x] 确认字段结构

### Step 1: 提取报纸名称列表

**目标**：获取所有唯一的报纸名称

**脚本**：`newspaper_extractor.py`

**功能**：
```python
def extract_newspaper_names():
    """提取所有唯一报纸名称"""
    # 遍历所有JSONL文件
    # 逐行读取source字段
    # 收集唯一名称
    # 保存到txt文件
```

**输出**：`newspaper_names.txt`（每行一个报纸名称）

**SLURM脚本**：`extract_newspapers.sh`
```bash
#!/bin/bash
#SBATCH --job-name=extract_newspapers
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/extract_newspapers_%j.out

source ~/.bash_profile
conda activate opinion
python newspaper_extractor.py extract_names
```

**运行命令**（用户在登录节点执行）：
```bash
sbatch extract_newspapers.sh
```

---

### Step 2: 构建报纸-省份映射

**目标**：建立报纸名称到省份的映射关系

**方法A：Wikipedia API（推荐）**
- 使用Wikipedia API搜索报纸名称
- 提取报纸所属地区信息
- 手动校验和补充

**方法B：人工整理**
- 直接根据报纸名称判断（如"北京日报"→"北京"）
- 参考中国报纸目录

**输出**：`newspaper_province_mapping.json`
```json
{
  "北京日报": "北京",
  "解放日报": "上海",
  "南方周末": "广东",
  ...
}
```

**注意**：此步骤需要网络访问，在登录节点或本地环境执行

---

### Step 3: 构建省级语料库

**目标**：按省份分组、清洗、分词报纸文本，生成训练语料

**脚本**：`newspaper_corpus_builder.py`

**功能**：
```python
class NewspaperCorpusBuilder:
    def build_provincial_corpus(mapping_file):
        """构建省级语料库"""
        # 1. 加载报纸-省份映射
        # 2. 初始化省级语料文件
        # 3. 逐文件、逐行处理：
        #    - 读取source和pdf_txt
        #    - 映射到省份
        #    - 清洗文本
        #    - 分词
        #    - 写入对应省份的语料文件
        # 4. 统计各省份文章数量
```

**文本清洗**：
- 去除特殊字符和格式标记（如""、"　　"）
- 去除URL
- 去除过短文本（<20字）

**分词**：
- 使用jieba分词
- 去除停用词
- 保存为空格分隔的token序列

**输出目录结构**：
```
newspaper_corpus/
├── 北京/
│   ├── corpus_000000
│   ├── corpus_000001
│   └── ...
├── 上海/
│   ├── corpus_000000
│   └── ...
└── ...
```

**SLURM脚本**：`build_corpus.sh`
```bash
#!/bin/bash
#SBATCH --job-name=build_newspaper_corpus
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/build_corpus_%j.out

source ~/.bash_profile
conda activate opinion
python newspaper_corpus_builder.py build
```

**运行命令**：
```bash
sbatch build_corpus.sh
```

---

### Step 4: 训练省级Word2Vec模型

**目标**：为每个省份训练Word2Vec模型

**脚本**：复用现有的`gender_embedding_trainer.py`，或创建`newspaper_embedding_trainer.py`

**参数**：
- vector_size: 300
- window: 5
- min_count: 20（根据数据量调整）
- workers: 8
- epochs: 10

**SLURM脚本**：`train_newspaper_embeddings.sh`
```bash
#!/bin/bash
#SBATCH --job-name=train_newspaper_embeddings
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_embeddings_%j.out

source ~/.bash_profile
conda activate opinion

# 方案1：训练所有省份
python newspaper_embedding_trainer.py train_all

# 方案2：分组训练（避免内存问题）
# python newspaper_embedding_trainer.py train --group 0
```

**输出**：
```
newspaper_embedding_models/
├── model_北京.model
├── model_上海.model
└── ...
```

---

### Step 5: 性别规范指数计算

**目标**：基于训练好的模型计算性别规范指数

**脚本**：复用`gender_norm_index_builder.py`

**步骤**：
1. OOV检查
2. 构建性别轴
3. 计算投影值
4. 计算WEAT效应量

---

## 文件清单

### Python脚本
1. `newspaper_extractor.py` - 提取报纸名称
2. `newspaper_corpus_builder.py` - 构建省级语料库
3. `newspaper_embedding_trainer.py` - 训练Word2Vec模型
4. （复用）`gender_norm_index_builder.py` - 计算性别规范指数

### SLURM脚本
1. `extract_newspapers.sh` - 提取报纸名称
2. `build_corpus.sh` - 构建语料库
3. `train_newspaper_embeddings.sh` - 训练模型

### 数据文件
1. `newspaper_names.txt` - 报纸名称列表
2. `newspaper_province_mapping.json` - 报纸-省份映射
3. `newspaper_corpus/` - 省级语料库目录
4. `newspaper_embedding_models/` - 模型输出目录

---

## 注意事项

### 1. 内存管理
- 永远不要将整个文件加载到内存
- 使用流式处理（逐行读取）
- 定期调用`gc.collect()`

### 2. 数据质量
- 检查`pdf_txt`字段是否为空
- 处理编码问题
- 去除重复文章

### 3. 报纸-省份映射
- 某些报纸可能是全国性报纸（如"人民日报"）
- 某些报纸可能无法确定省份
- 需要制定策略处理这些情况：
  - 方案A：排除全国性报纸
  - 方案B：根据发行地分配
  - 方案C：单独建立"全国"类别

### 4. 语料平衡
- 不同省份的报纸数量和文章数量可能差异很大
- 可能需要采样策略来平衡语料规模

### 5. SLURM资源申请
- 提取报纸名称：2小时，16G内存
- 构建语料库：24小时，64G内存
- 训练模型：48小时，64G内存（根据数据规模调整）

---

## 执行顺序

1. **[用户]** 在登录节点执行：`sbatch extract_newspapers.sh`
2. **[用户]** 等待任务完成，获取`newspaper_names.txt`
3. **[用户]** 在有网络环境执行报纸-省份映射
4. **[用户]** 上传映射文件到服务器
5. **[用户]** 在登录节点执行：`sbatch build_corpus.sh`
6. **[用户]** 等待语料库构建完成
7. **[用户]** 在登录节点执行：`sbatch train_newspaper_embeddings.sh`
8. **[用户]** 等待模型训练完成
9. **[用户]** 运行性别规范指数计算

---

## 与Weibo分析的差异

| 维度 | Weibo数据 | 报纸数据 |
|------|-----------|----------|
| 数据源 | 社交媒体 | 传统媒体 |
| 地理标识 | 用户定位 | 报纸归属 |
| 文本长度 | 短文本 | 长文本 |
| 时间跨度 | 2020/2024 | 待确定 |
| 语言风格 | 口语化 | 正式书面语 |
| 性别偏见来源 | 用户表达 | 媒体呈现 |

---

## 预期问题与解决方案

### 问题1：某些省份报纸数量过少
**解决**：
- 设置最小文章数阈值（如10万篇）
- 合并相邻省份
- 标记低置信度结果

### 问题2：全国性报纸的处理
**解决**：
- 初期排除
- 后期可单独分析对比

### 问题3：模型训练时间过长
**解决**：
- 分组训练
- 调整min_count参数
- 使用更少的epochs

---

## 下一步行动

1. ✅ 创建计划文档
2. ⏳ 编写`newspaper_extractor.py`
3. ⏳ 编写`extract_newspapers.sh`
4. ⏳ 等待用户提取报纸名称
5. ⏳ 构建报纸-省份映射
6. ⏳ 编写后续脚本...
