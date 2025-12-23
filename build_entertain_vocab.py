"""
构建娱乐账号词汇表
从指定年份热搜榜单数据中提取名词（覆盖明星、影视剧名、事件等）

功能说明:
1. 从bangdan数据中提取热搜词
2. 过滤广告（检查actionlog.ext中的ads_word字段）
3. 对热搜词去重
4. 使用jieba分词+词性标注，提取2-4个字符的名词
5. 按频率排序，输出前N个高频名词到txt文件（一行一个词）
6. 用户可进一步手工筛选得到娱乐相关词汇

使用方法:
---------
1. 探索数据结构（解压并查看第一个文件）:
   python build_entertain_vocab.py explore --year=2020

2. 构建娱乐词汇表（主要功能）:
   python build_entertain_vocab.py build --year=2020
   python build_entertain_vocab.py build --year=2020 --top_n=3000
   python build_entertain_vocab.py build --year=2020 --output_file=wordlists/my_nouns.txt

参数说明:
---------
- year: 年份（必需）
- top_n: 输出前N个高频词，默认5000
- output_file: 输出文件路径，默认为 wordlists/entertainment_nouns_{year}.txt

依赖安装:
---------
需要安装 jieba（中文分词工具）:
  pip install jieba

输出格式:
---------
输出文件每行一个名词，按频率从高到低排序
包括但不限于：人名、影视剧名、事件名、地点等
例如:
  王一博
  三十而已
  赵丽颖
  演唱会
  金鹰奖
  ...

用户可根据输出结果进一步筛选娱乐相关词汇
"""

import os
import re
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import fire

from configs.configs import ORIGIN_DATA_DIR
from utils.utils import extract_single_7z_file, extract_7z_files

# 导入jieba进行分词和词性标注
try:
    import jieba.posseg as pseg

    print("✓ 使用 jieba 进行中文分词和词性标注")
    JIEBA_AVAILABLE = True
except ImportError:
    print("❌ 未安装 jieba，请运行: pip install jieba")
    JIEBA_AVAILABLE = False


def get_bangdan_files_dir(year):
    return f"{ORIGIN_DATA_DIR}/{year}/bangdan/"


def get_bangdan_unzipped_files_dir(year):
    return f"bangdan_data/{year}/"


def unzip_all_bangdan_files(year):
    """
    将原始微博数据解压缩到当前目录的bangdan_data文件夹
    """
    bangdan_files_dir = get_bangdan_files_dir(year)
    unzipped_dir = get_bangdan_unzipped_files_dir(year)
    extract_7z_files(source_folder=bangdan_files_dir, target_folder=unzipped_dir)
    return True


def extract_nouns(text):
    """
    从文本中提取2-4个字符的名词

    使用jieba分词+词性标注，提取所有名词类词汇
    包括：人名、地名、机构名、作品名等

    Args:
        text: 输入文本

    Returns:
        list: 提取到的名词列表
    """
    if not JIEBA_AVAILABLE:
        print("❌ jieba未安装，无法提取名词")
        return []

    nouns = []

    try:
        # 使用jieba进行分词和词性标注
        words = pseg.cut(text)

        for word, flag in words:
            # 提取名词类词汇
            # jieba词性标注中，以'n'开头的都是名词：
            # - n: 普通名词
            # - nr: 人名
            # - nz: 其他专有名词
            # - ns: 地名
            # - nt: 机构团体名
            # - nw: 作品名
            # - nrfg: 人名 (复合)
            if flag in ["nr", "nrfg", "nw"] and 2 <= len(word) <= 4:
                # 确保是中文字符
                if all("\u4e00" <= char <= "\u9fff" for char in word):
                    nouns.append(word)
    except Exception as e:
        print(f"jieba处理出错: {e}")

    return nouns


def is_advertisement(actionlog_ext):
    """
    判断是否为广告

    Args:
        actionlog_ext: actionlog中的ext字段

    Returns:
        bool: True表示是广告
    """
    if actionlog_ext and "ads_word" in actionlog_ext:
        return True
    return False


def extract_hotwords_from_bangdan_file(file_path, verbose=False):
    """
    从单个bangdan文件中提取热搜词（过滤广告，提取名词）

    Args:
        file_path: bangdan文件路径
        verbose: 是否打印详细信息

    Returns:
        list: 提取到的名词列表
    """
    nouns_list = []
    hotwords_set = set()  # 用于去重热搜词
    ad_count = 0
    valid_count = 0

    if not os.path.exists(file_path):
        if verbose:
            print(f"⚠️  文件不存在: {file_path}")
        return []

    with open(file_path, "r", errors="replace") as rfile:
        for line in rfile.readlines():
            line = line.strip()
            if not line:
                continue

            line_data = line.split("\t")
            if len(line_data) < 2:
                continue

            try:
                data = json.loads(line_data[1])
            except json.JSONDecodeError:
                continue

            # 解析bangdan数据
            if "bangdan" not in data:
                continue

            try:
                bangdan_data = json.loads(data["bangdan"])
            except (json.JSONDecodeError, TypeError):
                continue

            if type(bangdan_data) is not dict:
                continue

            if "cards" not in bangdan_data or bangdan_data["cards"] is None:
                continue

            # 遍历所有card
            for card in bangdan_data["cards"]:
                if str(card.get("card_type")) != "11":
                    continue

                card_group = card.get("card_group", [])
                for s_card in card_group:
                    if str(s_card.get("card_type")) != "4":
                        continue

                    # 检查是否为广告
                    actionlog = s_card.get("actionlog", {})
                    actionlog_ext = actionlog.get("ext", "")

                    if is_advertisement(actionlog_ext):
                        ad_count += 1
                        continue

                    # 提取desc字段
                    desc = s_card.get("desc", "")
                    if not desc or len(desc) <= 1:
                        continue

                    # 去重
                    if desc in hotwords_set:
                        continue
                    hotwords_set.add(desc)

                    valid_count += 1

                    # 提取名词
                    nouns = extract_nouns(desc)
                    nouns_list.extend(nouns)

    if verbose:
        print(f"  文件: {os.path.basename(file_path)}")
        print(
            f"    有效热搜: {valid_count}, 过滤广告: {ad_count}, 提取名词: {len(nouns_list)}"
        )

    return nouns_list


def explore_bangdan_data(year: int):
    """
    探索bangdan数据结构
    解压一个文件，打印cardlistInfo和cards的前10个元素
    """
    bangdan_files_dir = get_bangdan_files_dir(year)
    unzipped_dir = get_bangdan_unzipped_files_dir(year)

    # 确保目标文件夹存在
    if not os.path.exists(unzipped_dir):
        os.makedirs(unzipped_dir)

    # 找到第一个.7z文件
    if not os.path.exists(bangdan_files_dir):
        print(f"目录不存在: {bangdan_files_dir}")
        return

    # 获取所有.7z文件
    zip_files = [f for f in os.listdir(bangdan_files_dir) if f.endswith(".7z")]
    if not zip_files:
        print(f"在 {bangdan_files_dir} 中没有找到.7z文件")
        return

    # 解压第一个文件
    first_zip_file = os.path.join(bangdan_files_dir, zip_files[0])
    print(f"正在解压文件: {first_zip_file}")
    result = extract_single_7z_file(
        file_path=first_zip_file, target_folder=unzipped_dir
    )

    if result != "success":
        print("解压失败")
        return

    # 找到解压后的文件
    unzipped_files = [f for f in os.listdir(unzipped_dir) if not f.endswith(".7z")]
    if not unzipped_files:
        print(f"解压后没有找到文件在 {unzipped_dir}")
        return

    # 读取第一个文件的第一行有效数据
    first_file = os.path.join(unzipped_dir, unzipped_files[0])
    print(f"\n正在读取文件: {first_file}")

    with open(first_file, "r", errors="replace") as rfile:
        for line_num, line in enumerate(rfile.readlines(), 1):
            line = line.strip()
            if not line:
                continue

            line_data = line.split("\t")
            if len(line_data) < 2:
                continue

            try:
                data = json.loads(line_data[1])
            except json.JSONDecodeError as e:
                continue

            # 解析bangdan数据
            if "bangdan" not in data:
                continue

            try:
                bangdan_data = json.loads(data["bangdan"])
            except (json.JSONDecodeError, TypeError):
                continue

            if type(bangdan_data) is not dict:
                continue

            print(f"\n{'='*80}")
            print(f"找到有效数据 (第 {line_num} 行)")
            print(f"{'='*80}\n")

            # 打印cardlistInfo
            if "cardlistInfo" in bangdan_data:
                print("=" * 80)
                print("cardlistInfo 数据格式:")
                print("=" * 80)
                print(
                    json.dumps(
                        bangdan_data["cardlistInfo"], ensure_ascii=False, indent=2
                    )
                )
                print()
            else:
                print("注意: bangdan_data 中没有 'cardlistInfo' 字段")
                print(f"bangdan_data 的键: {list(bangdan_data.keys())}")
                print()

            # 打印cards的前10个元素
            if "cards" in bangdan_data and bangdan_data["cards"]:
                print("=" * 80)
                print(
                    f"cards 数据格式 (前10个元素，共 {len(bangdan_data['cards'])} 个):"
                )
                print("=" * 80)
                for i, card in enumerate(bangdan_data["cards"][:10], 1):
                    print(f"\n--- Card {i} ---")
                    print(json.dumps(card, ensure_ascii=False, indent=2))
            else:
                print("注意: bangdan_data 中没有 'cards' 字段或 cards 为空")
                if "cards" in bangdan_data:
                    print(f"cards 类型: {type(bangdan_data['cards'])}")
                print()

            # 只处理第一个有效数据，然后停止
            print("\n" + "=" * 80)
            print("探索完成，程序停止")
            print("=" * 80)
            return

    print("没有找到有效的bangdan数据")


def build_entertainment_vocab(
    year: int, top_n: int = 5000, output_file: str = None, mode: str = "test"
):
    """
    构建娱乐词汇表：从bangdan数据中提取名词并按频率排序

    Args:
        year: 年份
        top_n: 输出前N个高频词，默认5000
        output_file: 输出文件路径，如果不指定则自动生成
    """
    if not JIEBA_AVAILABLE:
        print("❌ jieba未安装，无法执行。请运行: pip install jieba")
        return

    print(f"\n{'='*70}")
    print(f"开始构建 {year} 年娱乐词汇表（名词提取）")
    print(f"{'='*70}\n")

    # 设置输出文件
    if output_file is None:
        output_file = f"wordlists/entertainment_nouns_{year}.txt"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if mode != "test":
        unzip_all_bangdan_files(year)

    data_dir = get_bangdan_unzipped_files_dir(year)

    # 获取所有bangdan文件
    bangdan_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("weibo_bangdan.")
    ]

    if not bangdan_files:
        print(f"❌ 未找到bangdan文件在: {data_dir}")
        return

    bangdan_files.sort()
    print(f"✓ 找到 {len(bangdan_files)} 个bangdan文件\n")

    # 提取所有名词
    all_nouns = defaultdict(int)
    print("开始处理文件...")

    for i, file_path in enumerate(bangdan_files, 1):
        if i % 30 == 0 or i == 1:  # 每30个文件打印一次进度
            print(f"  进度: {i}/{len(bangdan_files)} ({i/len(bangdan_files)*100:.1f}%)")

        nouns = extract_hotwords_from_bangdan_file(file_path, verbose=False)
        for noun in nouns:
            all_nouns[noun] += 1

    print(f"\n✓ 处理完成！共提取 {len(all_nouns)} 个名词（含重复）\n")

    # 按频率排序
    sorted_nouns = sorted(all_nouns.items(), key=lambda x: x[1], reverse=True)
    sorted_nouns = sorted_nouns[:top_n]

    # 输出到文件
    with open(output_file, "w", encoding="utf-8") as f:
        for noun, count in sorted_nouns:
            f.write(f"{noun}\n")

    print(f"{'='*70}")
    print(f"✅ 词汇表已保存到: {output_file}")
    print(f"{'='*70}\n")

    # 打印统计信息
    print("📊 高频名词 Top 30:\n")
    print(f"{'排名':<6} {'名词':<10} {'频次':<10}")
    print("-" * 35)
    for i, (noun, count) in enumerate(sorted_nouns[:30], 1):
        print(f"{i:<6} {noun:<10} {count:<10}")

    print(f"\n{'='*70}")
    print(f"统计信息:")
    print(f"  总名词数（含重复）: {len(all_nouns):,}")
    print(f"  唯一名词数: {len(noun_counter):,}")
    print(f"  输出词汇数: {min(top_n, len(sorted_nouns)):,}")
    print(f"  最高频次: {sorted_nouns[0][1] if sorted_nouns else 0}")
    print(
        f"  最低频次（Top {top_n}）: {sorted_nouns[min(top_n-1, len(sorted_nouns)-1)][1] if sorted_nouns else 0}"
    )
    print(f"{'='*70}")
    print(f"\n💡 提示: 请手工审查输出文件，筛选出娱乐相关的名词\n")


class BangdanAnalyzer(object):

    def __init__(
        self,
        year: int,
    ):
        self.year = year
        self.data_dir = get_bangdan_unzipped_files_dir(year)
        self.bangdan_type = "1"

    def get_file_path(self, date: str = None):
        # date should be yyyy-mm-dd format
        return os.path.join(self.data_dir, f"weibo_bangdan.{date}")

    def get_bangdan_text_from_file(self, file_path: str, date: str):
        """
        一行bangdan信息的格式：timestamp,date,text,hot,rear
        例如：1111111111,2022-01-01,这是一个热搜话题,10000000,100
        """

        bangdan_text_list = []

        # 考虑file path是否存在
        if not os.path.exists(file_path):
            print(f"File not exists: {file_path}")
            return None
        with open(file_path, "r", errors="replace") as rfile:
            for line in rfile.readlines():
                line = line.strip()
                line_data = line.split("\t")
                if len(line_data) < 2:
                    print("line data cannot be splitted")
                    continue
                try:
                    data = json.loads(line_data[1])
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
                    # 打印出错误位置
                    print(f"Error at line {e.lineno}, column {e.colno}")
                    # 打印出错误字符位置
                    print(
                        f"Error at character {e.pos}, {line_data[1][int(e.pos)-20: int(e.pos)+20]}"
                    )
                    continue
                crawler_time_stamp = data["crawler_time_stamp"]
                if data["type"] != self.bangdan_type:
                    # print(f"wrong data type: {data['type']}")
                    # 排除不允许的榜单类型
                    # 不是实时榜
                    continue
                data = json.loads(data["bangdan"])
                if type(data) is not dict:
                    print(f"bad data type")
                    print(data)
                    continue
                if "cards" not in data.keys() or data["cards"] is None:
                    print(f"bad data type in file {file_path}")
                    continue
                for card in data["cards"]:
                    if str(card["card_type"]) != "11":
                        continue
                    card_group = card["card_group"]
                    for s_card in card_group:
                        if str(s_card["card_type"]) != "4":
                            continue
                        if "desc" in s_card.keys():
                            text = s_card["desc"]
                            if len(text) <= 5:
                                # 太短的话题丢掉
                                continue

                            hot = ""
                            if "desc_extr" in s_card.keys():
                                # 讨论小于10w的丢掉
                                # print(s_card["desc_extr"])
                                hot_number = re.findall(
                                    r"\d+", str(s_card["desc_extr"])
                                )
                                hot = hot_number[0] if len(hot_number) > 0 else None

                            is_rear = (
                                1
                                if re.search(self.rear_pattern, text) is not None
                                else 0
                            )

                            bangdan_text_list.append(
                                f"{crawler_time_stamp},{date},{text},{hot},{is_rear}"
                            )

                        else:
                            print(
                                f"desc not in keys! file_name {file_path}, data: {s_card}"
                            )
        return bangdan_text_list

    def analyze(self):
        # 遍历self.year的一整年的每一天 (通过datetime)
        for date in [datetime(self.year, 1, 1) + timedelta(days=i) for i in range(365)]:
            date_str = date.strftime("%Y-%m-%d")
            month_str = date.strftime("%Y-%m")
            file_path = self.get_file_path(date_str)
            if not os.path.exists(file_path):
                print(f"File not exists: {file_path}")
                continue
            bangdan_text_list = self.get_bangdan_text_from_file(file_path, date_str)
            if bangdan_text_list is None:
                continue
            with open(f"bangdan_working_data/{month_str}.csv", "a") as wfile:
                wfile.write("\n".join(bangdan_text_list))
                wfile.write("\n")
            print(f"processed {date_str} in year {self.year}")


if __name__ == "__main__":
    fire.Fire(
        {
            "explore": explore_bangdan_data,
            "build": build_entertainment_vocab,
            "analyze": BangdanAnalyzer,  # 保留旧的analyze功能
        }
    )
