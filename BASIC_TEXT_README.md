# 微博数据清洗脚本使用说明

## 功能概述

`basic_text_extractor.py` 用于清洗和处理原始微博数据：
- ✅ 只去重（按weibo_id）
- ✅ 不筛选关键词
- ✅ 不筛选用户
- ✅ 将所有数据整理成统一的parquet格式
- ✅ 按天存储到 `~/cleaned_weibo_data`
- ✅ 实时去重（在处理过程中即完成去重，节省内存）

## 使用方法

### 基本命令

```bash
# 处理指定年份的数据（位置参数）
python basic_text_extractor.py 2020 --mode 0  # 上半年
python basic_text_extractor.py 2020 --mode 1  # 下半年

# 也可以使用命名参数
python basic_text_extractor.py --year 2020 --mode 0
python basic_text_extractor.py --year 2020 --mode 1
```

### 参数说明

- `year`: 年份（位置参数，必需）
- `mode`: 处理模式（可选，默认：0）
  - `0`: 上半年（1月1日 - 6月30日）
  - `1`: 下半年（7月1日 - 12月31日）
- `check`: 只检查格式，不处理数据（可选，默认：False）

### 格式检查功能

**独立格式检查模式**：使用 `--check` 参数检查指定年份1月1日的数据格式

```bash
# 检查2023年数据格式
python basic_text_extractor.py 2023 --check

# 或使用命名参数
python basic_text_extractor.py --year 2023 --check
```

这会：
- 解压该年份1月1日的文件
- 打印前3行数据（每行前200字符）
- 不做进一步处理
- 用于检查新年度数据格式是否变化

## 数据格式支持

脚本支持三种数据格式：

1. **旧格式**（2019年8月9日之前）：Tab分隔的文本格式
2. **JSON格式**（2019年8月9日及之后）：Tab分隔，第二列是JSON
3. **特殊格式**（2020-06-30）：CSV格式

外壳会自动根据日期选择合适的处理函数。

## 输出格式

输出Parquet文件包含以下字段：

```python
{
    "weibo_id": "",           # 微博ID（用于去重）
    "user_id": "",            # 用户ID
    "is_retweet": "",         # 是否为转发
    "nick_name": "",          # 昵称
    "user_type": "",          # 用户类型
    "weibo_content": "",      # 微博内容
    "zhuan": "",              # 转发数
    "ping": "",               # 评论数
    "zan": "",                # 点赞数
    "device": "",             # 设备
    "locate": "",             # 定位信息
    "time_stamp": "",         # 时间戳
    "r_user_id": "",          # 被转发用户ID
    "r_nick_name": "",        # 被转发用户昵称
    "r_user_type": "",        # 被转发用户类型
    "r_weibo_id": "",         # 被转发微博ID
    "r_weibo_content": "",    # 被转发微博内容
    "r_zhuan": "",            # 被转发微博转发数
    "r_ping": "",             # 被转发微博评论数
    "r_zan": "",              # 被转发微博点赞数
    "r_device": "",           # 被转发微博设备
    "r_location": "",         # 被转发微博定位
    "r_time": "",             # 被转发时间
    "r_time_stamp": "",       # 被转发时间戳
    "src": "",                # 来源
    "tag": "",                # 标签
    "lat": "",                # 纬度
    "lon": "",                # 经度
}
```

## 日志文件

日志保存在 `logs/clean_weibo_log_{year}_{mode}.txt`

每条日志记录：
- 处理日期
- 耗时
- 记录数

## 性能优化

### 实时去重机制
- 在处理每个数据块时即完成去重
- 不需要收集所有数据后再去重
- 大幅减少内存占用
- 提高处理速度

## 注意事项

1. **文件规模**：由于文件很大，脚本会：
   - 检查解压文件是否已存在，避免重复解压
   - 处理完毕后自动删除解压文件
   - 使用分块处理（每块50万行）

2. **数据目录**：
   - 输入：`configs.configs.ORIGIN_DATA_DIR`
   - 输出：`~/cleaned_weibo_data`

3. **解压目录**：临时文件存储在 `text_working_data/{year}/testing/`

---

## 数据分析

### 分析脚本 `basic_text_analyzer.py`

基于清洗后的数据进行进一步分析。

#### 功能概述

1. **用户设备更换频率分析**
   - 生成用户每日设备表（稀疏表格）
   - 标记用户切换设备的时间
   - 计算平均更换频率和间隔天数
   
2. **转发官方媒体情况分析**
   - 分析转发官方媒体的用户行为
   - 可按性别分组统计（需要profile数据）

#### 使用方法

**分析单年数据：**
```bash
# 分析所有项目
python basic_text_analyzer.py year 2020

# 只分析设备更换
python basic_text_analyzer.py year 2020 device

# 只分析转发情况
python basic_text_analyzer.py year 2020 retweet
```

**分析多年数据：**
```bash
# 分析2020-2022年所有项目
python basic_text_analyzer.py years [2020,2021,2022]

# 分析多年设备更换情况
python basic_text_analyzer.py years [2020,2021,2022,2023] device
```

#### 输出结果

分析结果保存在 `analysis_results/` 目录（全部为parquet格式）：

- `device_daily_{year}.parquet` - 用户每日设备表（稀疏矩阵）
- `device_changes_{year}.parquet` - 设备切换分析结果
  - user_id, nick_name, total_changes, change_dates
  - avg_days_between_changes, min/max_days_between_changes
- `retweet_media_{year}.parquet` - 转发官方媒体统计
  - user_id, retweet_count

#### 配置官方媒体ID列表

在 `basic_text_analyzer.py` 中修改 `OFFICIAL_MEDIA_IDS` 变量：

```python
OFFICIAL_MEDIA_IDS = set([
    "1234567890",  # 人民日报
    "0987654321",  # 新华社
    # ... 添加更多官方媒体ID
])
```

#### 注意事项

- 设备更换分析需要全年数据，建议数据完整后再运行
- 转发分析需要先配置官方媒体ID列表
- 性别分析需要额外加载用户profile数据（待实现）

---

## Profile数据提取

### Profile提取脚本 `basic_profile_extractor.py`

提取所有用户的profile数据，无用户ID限制。

#### 功能概述

- 提取所有用户profile数据（不限用户ID）
- 支持按半年处理
- 自动解压和清理临时文件
- 保存为parquet格式

#### 使用方法

```bash
# 处理2020年profile数据
python basic_profile_extractor.py 2020 --mode 0  # 上半年
python basic_profile_extractor.py 2020 --mode 1  # 下半年

# 检查格式
python basic_profile_extractor.py 2023 --check
```

#### 输出格式

输出保存到 `cleaned_profile_data/` 目录，parquet格式包含以下字段：

- date, timestamp, user_id, nick_name, user_type
- gender, verified_type, verified_reason, description
- fans_number, weibo_number, type
- friends_count, favourites_count, created_at
- allow_all_comment, bi_followers_count
- location, province, city, ip_location

#### 配置说明

在脚本中修改 `get_zipped_profile_file` 函数中的路径：
```python
PROFILE_BASE_DIR = "/your/path/to/weibo-data"
```

#### 注意事项

- profile数据文件较大，脚本会自动管理解压和清理
- 输出格式与微博数据保持一致
- 可用于后续性别分析等应用

---

## 数据清洗与合并

### 合并脚本 `basic_weibo_merger.py`

清洗微博数据并合并profile信息

#### 功能概述

- 清洗微博数据（统一格式、去除NA）
- 合并profile数据（性别、地理位置等）
- 可选：删除水军记录

#### 使用方法

```bash
# 只清洗数据
python basic_weibo_merger.py clean 2020

# 只合并profile
python basic_weibo_merger.py merge 2020

# 一站式处理（推荐）
python basic_weibo_merger.py all 2020 --remove_shuijun
```

#### 参数说明

- `year`: 年份（必需）
- `month`: 月份（可选）
- `profile_path`: profile数据路径
- `remove_shuijun`: 是否删除水军

#### 输出结果

- 输出目录：`merged_weibo_data/{year}/`
- 合并字段：user_type, gender, location, province, city, ip_location

---

## Embedding分析

### 性别和职业词Embedding分析 `gender_embedding_analyzer.py`

分析不同省份性别讨论内容的embedding，计算职业词与性别词的夹角

#### 功能概述

- 按省份分组训练Word2Vec模型
- 计算性别词表（男性/女性）的平均向量
- 计算职业词与性别向量的夹角
- 比较不同省份的职业性别刻板印象

#### 词表配置

**性别词表：**
- 男性：男、男人、男性、帅哥、先生、爸爸、父亲、老公、兄弟等
- 女性：女、女人、女性、美女、女士、妈妈、母亲、老婆、姐妹等

**职业词表：** 医生、护士、教师、工程师、程序员、设计师、律师、会计、销售等26个职业

#### 使用方法

```bash
# 分析所有省份
python gender_embedding_analyzer.py 2020

# 只分析特定省份
python gender_embedding_analyzer.py 2020 --province 北京
python gender_embedding_analyzer.py 2020 --province 广东
```

#### 输出结果

**1. 分析结果 (`embedding_analysis_{year}.parquet`)**
包含每个省份的：
- 数据统计
- 每个职业与性别向量的余弦相似度

**2. 详细向量 (`detailed_vectors_{year}.parquet`)**
包含每个省份的：
- 男性词平均向量
- 女性词平均向量
- 性别差异向量

**3. Word2Vec模型 (`model_{year}_{province}.model`)**
每个省份的训练模型（可用于进一步分析）

#### 参数说明

- `year`: 年份（必需）
- `province`: 省份名称（可选），不指定则处理所有省份

#### 注意事项

- 需要至少1000条微博数据才会训练模型
- 文本预处理使用jieba分词
- 模型维度：100维，窗口：5，最小词频：5

---

## 内容相似性分析

### 官号与性别内容相似性分析 `content_similarity_analyzer.py`

分析男性和女性讨论内容与官方账号发布内容的相关性

#### 功能概述

- 提取官方账号发布的微博内容
- 分别提取男性和女性的讨论内容
- 使用Word2Vec训练embedding模型
- 计算性别内容与官号内容的余弦相似度
- 比较不同性别的差异

#### 使用方法

```bash
# 先添加官方账号ID（可选，也可以直接在脚本中配置）
python content_similarity_analyzer.py add_account --account_id 1234567890

# 添加多个账号
python content_similarity_analyzer.py add_account --account_id 111222333
python content_similarity_analyzer.py add_account --account_id 444555666

# 运行分析
python content_similarity_analyzer.py analyze 2020
```

#### 配置文件

官方账号ID会保存在 `configs/official_account_ids.json`，也可以在脚本中直接修改 `OFFICIAL_ACCOUNT_IDS` 变量。

#### 输出结果

**1. 分析结果 (`similarity_analysis_{year}.parquet`)**
包含：
- 官方账号、男性、女性的内容数量
- 男性与官号内容相似度
- 女性与官号内容相似度
- 性别差异（男性相似度 - 女性相似度）
- 各组的embedding向量

**2. Word2Vec模型 (`model_{year}.model`)**
使用所有文本训练的embedding模型

#### 参数说明

- `year`: 年份（必需）
- `account_id`: 官方账号ID（add_account命令使用）

#### 注意事项

- 需要至少100条官方账号内容才会进行分析
- 需要至少100条男性和女性内容
- 相似度范围为-1到1，值越大表示越相似
- 性别差异为正表示男性更接近官号内容，为负表示女性更接近

