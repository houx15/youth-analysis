"""
报纸省级语料库构建器

功能：
1. 读取报纸JSONL文件
2. 使用映射文件将报纸映射到省份
3. 清洗文本
4. 分词（jieba）
5. 按省份写入语料文件

使用方法：
    python newspaper_corpus_builder.py build
    python newspaper_corpus_builder.py build --max_files 10  # 测试模式
"""

import json
import os
import re
import jieba
from collections import defaultdict
from typing import Dict, Set
import fire
from tqdm import tqdm


DATA_DIR = "/lustre/home/2401111059/newspaper_data/pdf_txt"
MAPPING_FILE = "gender_norms/newspaper_data/newspaper_province_mapping.json"
OUTPUT_DIR = "gender_norms/newspaper_data/newspaper_corpus"
LOG_DIR = "gender_norms/newspaper_data/logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 停用词（与Weibo分析一致）
STOPWORDS = set([
    "的", "是", "了", "在", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "那", "我", "他",
    "她", "我们", "你们", "他们", "她们", "什么", "怎么", "这个", "那个",
    "可以", "因为", "所以", "但是", "而且", "或者", "如果", "虽然",
    "已经", "可能", "应该", "需要", "通过", "进行", "提出", "以及",
    "本报", "记者", "报道", "日前", "近日", "今天", "昨天", "今年",
])


def load_mapping() -> Dict[str, str]:
    """加载报纸-省份映射"""
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # 转换为 newspaper -> province 的字典
    mapping = {}
    for newspaper, info in mapping_data.items():
        province = info['province']
        # 只保留省级数据，排除全国性和行业报纸
        if province not in ['全国', '行业', '未知']:
            mapping[newspaper] = province
    
    print(f"✓ 加载映射: {len(mapping)} 个报纸 -> 省份")
    return mapping


def clean_text(text: str) -> str:
    """清洗报纸文本"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # 去除特殊格式字符
    text = re.sub(r'[　\s]+', ' ', text)  # 多个空格/全角空格 -> 单个空格
    text = re.sub(r'', '', text)  # 特殊字符
    text = re.sub(r'\s+', ' ', text)  # 多个空格 -> 单个空格
    
    # 去除URL
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # 去除特殊标记
    text = re.sub(r'\[.*?\]', '', text)  # [xxx]
    text = re.sub(r'（[^）]*）', '', text)  # （xxx）
    
    # 去除标点符号（保留中文和字母数字）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
    
    return text.strip()


def segment_text(text: str) -> list:
    """分词并去除停用词"""
    if not text or len(text) < 20:  # 过短文本
        return []
    
    # 分词
    words = jieba.lcut(text, HMM=True)
    
    # 过滤
    filtered = [
        word.strip()
        for word in words
        if word.strip()
        and word.strip() not in STOPWORDS
        and len(word.strip()) > 1  # 至少2个字符
        and not word.strip().isdigit()  # 排除纯数字
    ]
    
    return filtered


class ProvinceCorpusWriter:
    """省级语料写入器（滚动文件）"""
    
    def __init__(self, province: str, max_bytes: int = 1024 * 1024 * 1024):
        """
        Args:
            province: 省份名称
            max_bytes: 单个文件最大字节数（默认1GB）
        """
        self.province = province
        self.max_bytes = max_bytes
        self.province_dir = os.path.join(OUTPUT_DIR, province)
        os.makedirs(self.province_dir, exist_ok=True)
        
        self.index = 0
        self.bytes_written = 0
        self.total_lines = 0
        self._open_next()
    
    def _open_next(self):
        """打开下一个文件"""
        while True:
            filename = f"corpus_{self.index:06d}"
            filepath = os.path.join(self.province_dir, filename)
            if not os.path.exists(filepath):
                break
            self.index += 1
        
        self.file = open(filepath, 'w', buffering=8 * 1024 * 1024, encoding='utf-8')
        self.bytes_written = 0
    
    def write(self, words: list):
        """写入一行（词列表）"""
        if not words or len(words) < 5:  # 至少5个词
            return
        
        line = ' '.join(words) + '\n'
        
        # 检查是否需要切换文件
        if self.bytes_written + len(line) > self.max_bytes:
            self.file.close()
            self.index += 1
            self._open_next()
        
        self.file.write(line)
        self.bytes_written += len(line)
        self.total_lines += 1
    
    def close(self):
        """关闭文件"""
        self.file.close()
    
    def stats(self) -> dict:
        """返回统计信息"""
        return {
            'province': self.province,
            'total_lines': self.total_lines,
            'files_created': self.index + 1,
        }


def build_corpus(max_files: int = None, min_article_length: int = 50, resume: bool = True, file_list: str = None):
    """
    构建省级语料库
    
    Args:
        max_files: 最多处理多少个文件（None表示全部，用于测试）
        min_article_length: 文章最小字符数（默认50）
        resume: 是否增量处理（跳过已处理的文件）
        file_list: 指定要处理的文件列表（txt文件路径，每行一个文件名）
    """
    print(f"\n{'='*60}")
    print(f"📰 开始构建报纸省级语料库")
    print(f"{'='*60}\n")
    
    # 加载映射
    mapping = load_mapping()
    
    # 获取要处理的文件列表
    if file_list and os.path.exists(file_list):
        # 从文件列表读取
        with open(file_list, 'r', encoding='utf-8') as f:
            all_files = [line.strip() for line in f if line.strip()]
        print(f"📋 从文件列表读取: {len(all_files)} 个文件")
    else:
        # 获取所有文件
        all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
    
    # 增量处理：检查已处理的文件
    processed_files = set()
    if resume:
        checkpoint_file = os.path.join(LOG_DIR, 'processed_files.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                processed_files = set(json.load(f))
            print(f"🔄 增量模式: 已处理 {len(processed_files)} 个文件")
    
    # 过滤已处理的文件
    if resume and processed_files:
        all_files = [f for f in all_files if f not in processed_files]
        print(f"📂 剩余 {len(all_files)} 个文件待处理")
    else:
        print(f"📂 找到 {len(all_files)} 个文件待处理")
    
    if max_files:
        all_files = all_files[:max_files]
    
    print(f"📊 映射到 {len(set(mapping.values()))} 个省份\n")
    
    # 初始化省级写入器
    province_writers: Dict[str, ProvinceCorpusWriter] = {}
    
    # 统计信息
    stats = {
        'total_articles': 0,
        'province_articles': defaultdict(int),
        'skipped_articles': 0,
        'errors': 0,
    }
    
    # 处理每个文件
    for filename in tqdm(all_files, desc="处理文件"):
        filepath = os.path.join(DATA_DIR, filename)
        
        # 尝试不同编码
        file_handle = None
        successful_encoding = None
        encodings = ['utf-8', 'gb18030', 'gbk', 'gb2312', 'latin1']
        
        for encoding in encodings:
            try:
                file_handle = open(filepath, 'r', encoding=encoding)
                # 测试读取前几行
                for _ in range(10):
                    test_line = file_handle.readline()
                    if not test_line:
                        break
                file_handle.seek(0)  # 重置到文件开头
                successful_encoding = encoding
                break
            except (UnicodeDecodeError, UnicodeError):
                if file_handle:
                    file_handle.close()
                    file_handle = None
                continue
        
        if file_handle is None:
            # 所有编码都失败，使用utf-8 with errors='ignore'跳过无效字节
            file_handle = open(filepath, 'r', encoding='utf-8', errors='ignore')
            successful_encoding = 'utf-8 (ignoring errors)'
        
        try:
            with file_handle as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        article = json.loads(line)
                        source = article.get('source', '').strip()
                        text = article.get('pdf_txt', '')
                        
                        # 映射省份
                        if source not in mapping:
                            stats['skipped_articles'] += 1
                            continue
                        
                        province = mapping[source]
                        
                        # 清洗文本
                        cleaned = clean_text(text)
                        if len(cleaned) < min_article_length:
                            stats['skipped_articles'] += 1
                            continue
                        
                        # 分词
                        words = segment_text(cleaned)
                        if len(words) < 5:
                            stats['skipped_articles'] += 1
                            continue
                        
                        # 初始化写入器（如果需要）
                        if province not in province_writers:
                            province_writers[province] = ProvinceCorpusWriter(province)
                        
                        # 写入
                        province_writers[province].write(words)
                        
                        stats['total_articles'] += 1
                        stats['province_articles'][province] += 1
                        
                    except json.JSONDecodeError as e:
                        # 只记录前几次错误
                        if stats['errors'] < 5:
                            print(f"\n⚠️  JSON解析错误 {filename}:{line_num}: {str(e)[:100]}")
                        stats['errors'] += 1
                        continue
                    except Exception as e:
                        stats['errors'] += 1
                        continue
            
            # 标记文件已处理
            processed_files.add(filename)
            
            # 每100个文件保存一次进度
            if len(processed_files) % 100 == 0 and resume:
                checkpoint_file = os.path.join(LOG_DIR, 'processed_files.json')
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(list(processed_files), f, ensure_ascii=False)
        
        except Exception as e:
            print(f"\n❌ 处理文件 {filename} 失败: {e}")
            stats['errors'] += 1
            continue
    
    # 关闭所有写入器
    print(f"\n\n💾 保存语料库...")
    writer_stats = []
    for province, writer in province_writers.items():
        writer.close()
        writer_stats.append(writer.stats())
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"✅ 语料库构建完成")
    print(f"{'='*60}")
    print(f"  总文章数: {stats['total_articles']:,}")
    print(f"  跳过文章: {stats['skipped_articles']:,}")
    print(f"  错误数: {stats['errors']:,}")
    print(f"  省份数: {len(province_writers)}")
    
    print(f"\n📊 各省份文章数（Top 10）:")
    sorted_provinces = sorted(
        stats['province_articles'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for i, (province, count) in enumerate(sorted_provinces[:10], 1):
        print(f"  {i}. {province}: {count:,} 篇")
    
    # 保存统计信息
    stats_file = os.path.join(LOG_DIR, 'corpus_build_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_articles': stats['total_articles'],
            'province_articles': dict(stats['province_articles']),
            'writer_stats': writer_stats,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 统计信息已保存: {stats_file}")
    
    # 保存已处理文件列表（用于增量处理）
    if resume:
        checkpoint_file = os.path.join(LOG_DIR, 'processed_files.json')
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(list(processed_files), f, ensure_ascii=False)
        print(f"💾 已处理文件列表已保存: {checkpoint_file}")


if __name__ == "__main__":
    import pandas as pd  # 用于 isna 检查
    fire.Fire({'build': build_corpus})
