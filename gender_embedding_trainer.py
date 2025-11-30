"""
性别词的Word2Vec模型训练器

功能：
1. 按省份分组数据训练Word2Vec模型
2. 保存训练好的模型供后续分析使用

输入数据：
- 2020: cleaned_weibo_cov/{year}/ 下的parquet文件
- 2024: weibo_data_2024/ 下的parquet文件（需先prepare）
输出：embedding_models/{year}/ 下的模型文件
"""

import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
import jieba
import fire
import warnings
import glob
import re
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

DATA_DIR_2020 = "cleaned_weibo_cov"
DATA_DIR_2024 = "../cleaned_weibo_data"
PREPARED_DIR_2024 = "gender_embedding/prepared_weibo_2024"
OUTPUT_DIR = "gender_embedding/embedding_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREPARED_DIR_2024, exist_ok=True)

# 省份编码映射（GB/T 2260 中华人民共和国行政区划代码）
PROVINCE_CODE_TO_NAME = {
    "11": "北京",
    "12": "天津",
    "13": "河北",
    "14": "山西",
    "15": "内蒙古",
    "21": "辽宁",
    "22": "吉林",
    "23": "黑龙江",
    "31": "上海",
    "32": "江苏",
    "33": "浙江",
    "34": "安徽",
    "35": "福建",
    "36": "江西",
    "37": "山东",
    "41": "河南",
    "42": "湖北",
    "43": "湖南",
    "44": "广东",
    "45": "广西",
    "46": "海南",
    "50": "重庆",
    "51": "四川",
    "52": "贵州",
    "53": "云南",
    "54": "西藏",
    "61": "陕西",
    "62": "甘肃",
    "63": "青海",
    "64": "宁夏",
    "65": "新疆",
    "71": "台湾",
    "81": "香港",
    "82": "澳门",
    "100": "未知",
    "400": "未知",
}

# 反向映射：省份名称到编码
PROVINCE_NAME_TO_CODE = {v: k for k, v in PROVINCE_CODE_TO_NAME.items() if v != "未知"}

# 通用停用词
STOPWORDS = set(
    [
        "的",
        "是",
        "了",
        "在",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "那",
        "我",
        "他",
        "她",
        "我们",
        "你们",
        "他们",
        "她们",
        "什么",
        "怎么",
        "这个",
        "那个",
    ]
)


def clean_weibo_content_2024(text):
    if pd.isna(text) or text == "":
        return ""

    text = str(text)

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"t\.cn/\S+", "", text)
    text = re.sub(r"@\S+", "", text)

    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    return text.strip()


def clean_region_name_2024(region):
    if pd.isna(region) or region == "":
        return None

    region = str(region).strip()

    if region.startswith("发布于 "):
        region = region[4:].strip()

    return region if region else None


def preprocess_text(text):
    if pd.isna(text) or text == "":
        return []

    text = str(text)
    words = jieba.cut(text)
    words = [
        w.strip()
        for w in words
        if w.strip() and w not in STOPWORDS and len(w.strip()) > 1
    ]
    return words


def prepare_2024_data_by_month_group(year, start_month, end_month):
    print(f"\n{'='*60}")
    print(f"📦 准备 {year} 年 {start_month}-{end_month} 月数据")
    print(f"{'='*60}\n")

    start_date = datetime(year, start_month, 1)
    if end_month == 12:
        end_date = datetime(year, 12, 31)
    else:
        end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)

    date_range = [
        start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)
    ]

    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"\n处理日期: {date_str}")

        pattern = os.path.join(DATA_DIR_2024, f"{date_str}.parquet.temp_*")
        print(pattern)
        temp_files = sorted(glob.glob(pattern))

        if not temp_files:
            print(f"  ⚠️  未找到文件: {date_str}")
            continue

        print(f"  📂 找到 {len(temp_files)} 个临时文件")

        province_data = defaultdict(list)

        for temp_file in temp_files:
            try:
                df = pd.read_parquet(temp_file)

                if "weibo_content" not in df.columns or "region_name" not in df.columns:
                    print(f"  ⚠️  跳过文件 {temp_file}，缺少必要列")
                    continue

                df = df.dropna(subset=["weibo_id", "weibo_content", "region_name"])

                df["weibo_content"] = df["weibo_content"].apply(
                    clean_weibo_content_2024
                )
                # 如果weibo_content长度小于5，删除
                df = df[df["weibo_content"].apply(len) > 5]
                df["province"] = df["region_name"].apply(clean_region_name_2024)

                df = df.dropna(subset=["province"])
                # df = df[df["weibo_content"] != ""]

                df = df[["weibo_id", "weibo_content", "province"]]

                for province in df["province"].unique():
                    province_df = df[df["province"] == province].copy()
                    province_data[province].append(province_df)

                del df

            except Exception as e:
                print(f"  ❌ 读取文件 {temp_file} 失败: {e}")
                continue

        for province, data_list in province_data.items():
            if not data_list:
                continue

            combined_data = pd.concat(data_list, ignore_index=True)
            combined_data = combined_data.drop_duplicates(subset=["weibo_id"])
            del data_list

            province_dir = os.path.join(PREPARED_DIR_2024, province)
            os.makedirs(province_dir, exist_ok=True)

            output_file = os.path.join(province_dir, f"{date_str}.parquet")
            combined_data.to_parquet(output_file, engine="fastparquet", index=False)

            print(f"  ✓ {province}: {len(combined_data):,} 条 → {output_file}")
            del combined_data

        import gc

        gc.collect()

    print(f"\n✅ {start_month}-{end_month} 月数据准备完成")


def load_single_province_2024(province):
    province_dir = os.path.join(PREPARED_DIR_2024, province)

    if not os.path.exists(province_dir):
        return None

    pattern = os.path.join(province_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        return None

    data_list = []
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            data_list.append(df)
        except Exception as e:
            print(f"  ❌ 读取失败: {file_path} - {e}")
            continue

    if not data_list:
        return None

    combined_data = pd.concat(data_list, ignore_index=True)
    del data_list

    combined_data = combined_data.drop_duplicates(subset=["weibo_id"])

    if len(combined_data) < 100000:
        print(f"  ✗ {province}: {len(combined_data):,} 条 (数据量不足)")
        del combined_data
        return None

    return combined_data


def load_data_by_province_2020(year):
    year_dir = os.path.join(DATA_DIR_2020, str(year))
    if not os.path.exists(year_dir):
        print(f"❌ 未找到 {year} 年的数据目录")
        return None

    pattern = os.path.join(year_dir, "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        print(f"❌ 未找到 {year} 年的数据文件")
        return None

    print(f"📂 找到 {len(parquet_files)} 个文件，开始加载...")

    required_columns = ["weibo_content"]
    province_col = "province"
    required_columns.append("province")

    data_by_province = defaultdict(list)

    for file_idx, file_path in enumerate(parquet_files):
        try:
            df = pd.read_parquet(file_path, columns=required_columns)
            df = df.dropna(subset=[province_col, "weibo_content"])

            def convert_province_code(code):
                if pd.isna(code):
                    return None
                if isinstance(code, (int, float)):
                    code_str = str(int(code))
                else:
                    code_str = str(code).strip()

                if code_str in PROVINCE_CODE_TO_NAME:
                    return PROVINCE_CODE_TO_NAME[code_str]
                elif code_str in PROVINCE_NAME_TO_CODE.values():
                    return code_str
                else:
                    if not hasattr(convert_province_code, "_warned_codes"):
                        convert_province_code._warned_codes = set()
                    if code_str not in convert_province_code._warned_codes:
                        print(f"  ⚠️  发现未知省份编码: {code_str}，将保留原值")
                        convert_province_code._warned_codes.add(code_str)
                    return code_str

            df[province_col] = df[province_col].apply(convert_province_code)
            df = df.dropna(subset=[province_col])

            for province in df[province_col].unique():
                province_data = df[df[province_col] == province].copy()
                data_by_province[province].append(province_data)

            del df

            if (file_idx + 1) % 10 == 0:
                print(f"  已处理 {file_idx + 1}/{len(parquet_files)} 个文件...")

        except Exception as e:
            print(f"❌ 读取文件 {file_path} 失败: {e}")
            continue

    print(f"\n📊 按省份分组，共 {len(data_by_province)} 个省份")

    converted_provinces = sorted(data_by_province.keys())
    print(
        f"  转换后的省份列表: {', '.join(converted_provinces[:10])}{'...' if len(converted_provinces) > 10 else ''}"
    )

    result = {}
    for province, data_list in data_by_province.items():
        combined_data = pd.concat(data_list, ignore_index=True)
        del data_list

        if len(combined_data) > 1000:
            result[province] = combined_data
            print(f"  ✓ {province}: {len(combined_data):,} 条数据")
        else:
            print(f"  ✗ {province}: {len(combined_data):,} 条数据 (跳过，数据量不足)")
            del combined_data

    return result


def train_word2vec(texts, vector_size=300, window=5, min_count=20, workers=None):
    """
    训练Word2Vec模型（内存优化版本）

    参数调整说明：
    - vector_size: 300（更大的向量维度，更好的语义表达）
    - window: 5（上下文窗口）
    - min_count: 20（词频阈值，根据数据量调整）
    - workers: 线程数，None则自动设置为CPU核心数-1
    """
    if not texts or len(texts) < 100:
        return None

    # 自动设置workers，避免超过CPU核心数
    if workers is None:
        import multiprocessing

        workers = max(1, multiprocessing.cpu_count() - 1)

    # 限制workers数量，避免内存过度占用
    workers = min(workers, 8)

    model = Word2Vec(
        sentences=texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=10,
        sg=1,  # Skip-gram
        negative=10,  # 负采样
        seed=42,  # 可重复性
        max_vocab_size=None,
    )

    return model


def train_single_province(province, data, year):
    print(f"\n{'='*60}")
    print(f"🔧 训练省份: {province}")
    print(f"{'='*60}")

    data_count = len(data)
    print(f"  📊 原始数据: {data_count:,} 条")

    texts = []
    for row in data.itertuples():
        words = preprocess_text(row.weibo_content)
        if len(words) > 3:
            texts.append(words)

    del data
    import gc

    gc.collect()

    if len(texts) < 100000:
        print(f"  ❌ 文本量不足 ({len(texts)} 条)，跳过")
        del texts
        return None

    print(f"  📝 有效文本: {len(texts):,} 条")

    print(f"  🔧 训练Word2Vec模型...")
    model = train_word2vec(texts)
    if model is None:
        print(f"  ❌ 训练模型失败")
        del texts
        return None

    vocab_size = len(model.wv)
    print(f"  ✓ 模型训练完成，词汇表大小: {vocab_size:,}")

    stats = {
        "province": province,
        "data_count": data_count,
        "text_count": len(texts),
        "vocab_size": vocab_size,
    }

    del texts
    gc.collect()

    year_output_dir = os.path.join(OUTPUT_DIR, str(year))
    os.makedirs(year_output_dir, exist_ok=True)

    model_path = os.path.join(year_output_dir, f"model_{province}.model")
    model.save(model_path)
    print(f"  💾 模型已保存: {model_path}")

    del model
    gc.collect()

    return stats


def train_models_by_province_2020(data_by_province, year, province_filter=None):
    training_stats = []

    for province, data in data_by_province.items():
        if province_filter and province != province_filter:
            continue

        stats = train_single_province(province, data, year)
        if stats:
            training_stats.append(stats)

    return training_stats


def train_models_by_province_2024(provinces, year):
    training_stats = []

    for province in provinces:
        print(f"\n📂 加载省份: {province}")
        data = load_single_province_2024(province)

        if data is None:
            print(f"  ⚠️  跳过 {province}，无法加载数据")
            continue

        print(f"  ✓ 加载成功: {len(data):,} 条数据")

        stats = train_single_province(province, data, year)
        if stats:
            training_stats.append(stats)

    return training_stats


def save_training_stats(training_stats, year):
    """保存训练统计信息"""
    if not training_stats:
        print("❌ 没有训练统计信息")
        return

    stats_df = pd.DataFrame(training_stats)
    stats_file = os.path.join(OUTPUT_DIR, f"training_stats_{year}.csv")
    stats_df.to_csv(stats_file, index=False, encoding="utf-8-sig")
    print(f"\n✓ 训练统计信息已保存: {stats_file}")


PROVINCE_GROUPS = [
    ["北京", "天津", "河北", "山西"],
    ["内蒙古", "辽宁", "吉林", "黑龙江"],
    ["上海", "江苏", "浙江", "安徽"],
    ["福建", "江西", "山东"],
    ["河南", "湖北", "湖南"],
    ["广东", "广西", "海南"],
    ["重庆", "四川", "贵州", "云南"],
    ["西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"],
]


def prepare(year: int, start_month: int, end_month: int):
    if year != 2024:
        print("❌ prepare命令仅支持2024年数据")
        return

    prepare_2024_data_by_month_group(year, start_month, end_month)


def train(year: int, province: str = None, group: int = None):
    print(f"\n{'='*60}")
    print(f"🚀 开始训练 {year} 年数据的Word2Vec模型")
    print(f"{'='*60}\n")

    if year == 2020:
        data_by_province = load_data_by_province_2020(year)
        if not data_by_province:
            print("❌ 无法加载数据")
            return

        if province:
            if province not in data_by_province:
                print(f"❌ 未找到省份: {province}")
                print(f"可用省份: {', '.join(data_by_province.keys())}")
                return
            print(f"🎯 只训练省份: {province}\n")

        training_stats = train_models_by_province_2020(data_by_province, year, province)

    elif year == 2024:
        if group is not None:
            if group < 0 or group >= len(PROVINCE_GROUPS):
                print(
                    f"❌ 无效的分组编号: {group}，有效范围: 0-{len(PROVINCE_GROUPS)-1}"
                )
                return
            provinces = PROVINCE_GROUPS[group]
            print(f"🎯 训练分组 {group}: {', '.join(provinces)}\n")
        elif province:
            provinces = [province]
            print(f"🎯 只训练省份: {province}\n")
        else:
            print("❌ 2024年数据必须指定 --group 或 --province")
            return

        training_stats = train_models_by_province_2024(provinces, year)

    else:
        print(f"❌ 不支持的年份: {year}，目前支持2020和2024")
        return

    save_training_stats(training_stats, year)

    print(f"\n{'='*60}")
    print(f"🎉 {year} 年Word2Vec模型训练完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    fire.Fire({"prepare": prepare, "train": train})
