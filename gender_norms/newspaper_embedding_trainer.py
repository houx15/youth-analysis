"""
报纸省级Word2Vec模型训练器

功能：
1. 从已构建的省级语料库训练Word2Vec模型
2. 按5组省份并行训练

使用方法：
    python newspaper_embedding_trainer.py train --group 0
    python newspaper_embedding_trainer.py train --province 北京

语料库位置：gender_norms/newspaper_data/newspaper_corpus/{省份}/
模型输出：gender_norms/newspaper_data/embedding_models/
"""

import gc
import glob
import os
from typing import List, Optional

import fire
from gensim.models import Word2Vec
from tqdm import tqdm

CORPUS_DIR = "gender_norms/newspaper_data/newspaper_corpus"
OUTPUT_DIR = "gender_norms/newspaper_data/embedding_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROVINCE_GROUPS = [
    ["北京", "天津", "河北", "山西", "内蒙古", "辽宁"],
    ["吉林", "黑龙江", "上海", "江苏", "浙江", "安徽"],
    ["福建", "江西", "山东", "河南", "湖北", "湖南"],
    ["广东", "广西", "海南", "重庆", "四川", "贵州"],
    ["云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆"],
]


class ProvinceCorpus:
    """省级语料库迭代器"""

    def __init__(self, province: str):
        self.province_dir = os.path.join(CORPUS_DIR, province)
        self.corpus_files = sorted(
            glob.glob(os.path.join(self.province_dir, "corpus_*"))
        )

    def __iter__(self):
        for filepath in self.corpus_files:
            with open(filepath, "r", buffering=8 * 1024 * 1024, encoding="utf-8") as f:
                for line in f:
                    words = line.strip().split()
                    if words:
                        yield words


def train_word2vec(
    corpus,
    vector_size: int = 300,
    window: int = 5,
    min_count: int = 20,
    workers: Optional[int] = None,
):
    """
    训练Word2Vec模型

    参数：
        vector_size: 向量维度
        window: 上下文窗口
        min_count: 词频阈值
        workers: 线程数
    """
    if workers is None:
        import multiprocessing

        workers = max(1, multiprocessing.cpu_count() - 1)

    workers = min(workers, 16)

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=10,
        sg=1,
        negative=10,
        seed=42,
        max_vocab_size=None,
    )

    model.wv.fill_norms()
    return model


def train_single_province(province: str, force_retrain: bool = False):
    """训练单个省份的模型"""
    print(f"\n{'='*60}")
    print(f"🔧 训练省份: {province}")
    print(f"{'='*60}")

    model_path = os.path.join(OUTPUT_DIR, f"model_{province}.kv")

    if not force_retrain and os.path.exists(model_path):
        print(f"  ⏭️  模型已存在: {model_path}")
        try:
            from gensim.models import KeyedVectors

            wv = KeyedVectors.load(model_path)
            vocab_size = len(wv)
            print(f"  ✓ 词汇表大小: {vocab_size:,}")
            return {"province": province, "vocab_size": vocab_size}
        except Exception as e:
            print(f"  ⚠️  加载失败: {e}，重新训练")

    corpus = ProvinceCorpus(province)

    if not corpus.corpus_files:
        print(f"  ❌ 未找到语料文件")
        return None

    print(f"  📂 语料文件: {len(corpus.corpus_files)} 个")

    model = train_word2vec(corpus)
    if model is None:
        print(f"  ❌ 训练失败")
        return None

    vocab_size = len(model.wv)
    print(f"  ✓ 训练完成，词汇表大小: {vocab_size:,}")

    model.wv.save(model_path)
    print(f"  💾 模型已保存: {model_path}")

    del model
    gc.collect()

    return {"province": province, "vocab_size": vocab_size}


def train(
    group: Optional[int] = None,
    province: Optional[str] = None,
    force_retrain: bool = False,
):
    """
    训练Word2Vec模型

    Args:
        group: 分组编号 (0-4)
        province: 单个省份名称
        force_retrain: 强制重新训练
    """
    print(f"\n{'='*60}")
    print(f"🚀 报纸省级Word2Vec模型训练")
    print(f"{'='*60}\n")

    provinces: List[str] = []

    if group is not None:
        if group < 0 or group >= len(PROVINCE_GROUPS):
            print(f"❌ 无效分组: {group}，有效范围: 0-{len(PROVINCE_GROUPS)-1}")
            return
        provinces = PROVINCE_GROUPS[group]
        print(f"📋 分组 {group}: {', '.join(provinces)}\n")
    elif province:
        provinces = [province]
        print(f"🎯 单省份: {province}\n")
    else:
        print("❌ 请指定 --group 或 --province")
        return

    training_stats = []

    for prov in provinces:
        stats = train_single_province(prov, force_retrain)
        if stats:
            training_stats.append(stats)

    if training_stats:
        print(f"\n{'='*60}")
        print(f"✅ 训练完成统计")
        print(f"{'='*60}")
        for stat in training_stats:
            print(f"  {stat['province']}: 词汇表 {stat['vocab_size']:,}")

    print(f"\n{'='*60}")
    print(f"🎉 训练完成！")
    print(f"{'='*60}\n")


def list_provinces():
    """列出所有可用省份"""
    print("\n📋 可用省份分组：\n")
    for i, group in enumerate(PROVINCE_GROUPS):
        print(f"  分组 {i}: {', '.join(group)}")

    print("\n📂 语料库中的省份：")
    provinces = sorted(os.listdir(CORPUS_DIR))
    print(f"  {', '.join(provinces)}")


if __name__ == "__main__":
    fire.Fire({"train": train, "list": list_provinces})
