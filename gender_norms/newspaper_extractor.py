"""
报纸名称提取脚本

功能：
1. 从所有报纸JSONL文件中提取唯一的报纸名称
2. 统计每个报纸的文章数量
3. 保存结果到txt和csv文件

使用方法：
    python newspaper_extractor.py extract_names
    python newspaper_extractor.py extract_names --sample_size 1000  # 只采样前1000行测试
"""

import json
import os
from collections import defaultdict
from typing import Dict, Set
import fire
from tqdm import tqdm


DATA_DIR = "/lustre/home/2401111059/newspaper_data/pdf_txt"
OUTPUT_DIR = "gender_norms/newspaper_data"


def extract_newspaper_names(sample_size: int = None):
    """
    从所有JSONL文件中提取唯一的报纸名称
    
    Args:
        sample_size: 如果提供，只处理每个文件的前sample_size行（用于测试）
    """
    print(f"\n{'='*60}")
    print(f"📰 开始提取报纸名称")
    print(f"{'='*60}\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取所有JSONL文件
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
    print(f"📂 找到 {len(all_files)} 个文件")
    
    # 统计报纸名称和文章数量
    newspaper_counts: Dict[str, int] = defaultdict(int)
    total_articles = 0
    errors = 0
    
    print(f"\n开始处理文件...\n")
    
    for filename in tqdm(all_files, desc="处理文件"):
        filepath = os.path.join(DATA_DIR, filename)
        
        # 尝试不同编码，使用errors='ignore'跳过无效字节
        encodings = ['utf-8', 'gb18030', 'gbk', 'gb2312', 'latin1']
        file_handle = None
        
        for encoding in encodings:
            try:
                file_handle = open(filepath, 'r', encoding=encoding)
                # 测试读取前几行
                for _ in range(10):
                    test_line = file_handle.readline()
                    if not test_line:
                        break
                file_handle.seek(0)  # 重置到文件开头
                break
            except (UnicodeDecodeError, UnicodeError):
                if file_handle:
                    file_handle.close()
                    file_handle = None
                continue
        
        if file_handle is None:
            # 所有编码都失败，使用utf-8 with errors='ignore'跳过无效字节
            file_handle = open(filepath, 'r', encoding='utf-8', errors='ignore')
        
        try:
            with file_handle as f:
                line_count = 0
                for line in f:
                    # 如果设置了sample_size，只处理前N行
                    if sample_size and line_count >= sample_size:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        article = json.loads(line)
                        source = article.get('source', '').strip()
                        
                        if source:
                            newspaper_counts[source] += 1
                            total_articles += 1
                        
                        line_count += 1
                        
                    except json.JSONDecodeError as e:
                        errors += 1
                        continue
                        
        except Exception as e:
            print(f"\n❌ 处理文件 {filename} 失败: {e}")
            errors += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ 提取完成")
    print(f"{'='*60}")
    print(f"  总文章数: {total_articles:,}")
    print(f"  唯一报纸数: {len(newspaper_counts):,}")
    print(f"  错误数: {errors:,}")
    
    # 按文章数量排序
    sorted_newspapers = sorted(newspaper_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 保存报纸名称列表（txt格式）
    names_file = os.path.join(OUTPUT_DIR, "newspaper_names.txt")
    with open(names_file, 'w', encoding='utf-8') as f:
        for name, count in sorted_newspapers:
            f.write(f"{name}\n")
    print(f"\n💾 报纸名称列表已保存: {names_file}")
    
    # 保存带统计的CSV文件
    stats_file = os.path.join(OUTPUT_DIR, "newspaper_stats.csv")
    with open(stats_file, 'w', encoding='utf-8-sig') as f:
        f.write("报纸名称,文章数量\n")
        for name, count in sorted_newspapers:
            # CSV转义：如果包含逗号或引号，需要用引号包裹
            if ',' in name or '"' in name:
                name = '"' + name.replace('"', '""') + '"'
            f.write(f"{name},{count}\n")
    print(f"💾 报纸统计已保存: {stats_file}")
    
    # 打印前20个报纸（预览）
    print(f"\n📊 前20个报纸（按文章数排序）:")
    print(f"{'序号':<6} {'报纸名称':<30} {'文章数量':>10}")
    print("-" * 50)
    for i, (name, count) in enumerate(sorted_newspapers[:20], 1):
        print(f"{i:<6} {name:<30} {count:>10,}")
    
    # 检查是否有潜在的全国性报纸
    national_keywords = ['人民', '光明', '经济', '参考', '环球', '中国']
    print(f"\n🔍 潜在的全国性报纸:")
    for name, count in sorted_newspapers:
        if any(keyword in name for keyword in national_keywords):
            print(f"  - {name}: {count:,} 篇文章")
    
    return sorted_newspapers


def preview_file(filename: str = None, n_lines: int = 5):
    """
    预览某个文件的前N行
    
    Args:
        filename: 文件名（不指定则预览第一个文件）
        n_lines: 预览行数
    """
    if filename is None:
        all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
        if not all_files:
            print("❌ 未找到文件")
            return
        filename = all_files[0]
    
    filepath = os.path.join(DATA_DIR, filename)
    print(f"\n📖 预览文件: {filename}\n")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            
            try:
                article = json.loads(line.strip())
                print(f"--- 第 {i+1} 篇文章 ---")
                print(f"报纸: {article.get('source', 'N/A')}")
                print(f"标题: {article.get('title', 'N/A')}")
                print(f"内容长度: {len(article.get('pdf_txt', ''))} 字符")
                print(f"URL: {article.get('pdf_url', 'N/A')}")
                print()
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析错误: {e}\n")


if __name__ == "__main__":
    fire.Fire({
        'extract_names': extract_newspaper_names,
        'preview': preview_file,
    })
