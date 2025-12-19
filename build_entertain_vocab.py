"""
构建娱乐账号词汇表
从指定年份热搜里面的娱乐榜加载文本
提取名词
词频最高的1000词输出为txt文件
一行一个词
"""

import os
import re
import json
from datetime import datetime, timedelta
import fire

from configs.configs import ORIGIN_DATA_DIR
from utils.utils import extract_single_7z_file


def get_bangdan_files_dir(year):
    return f"{ORIGIN_DATA_DIR}/{year}/bangdan/"


def get_bangdan_unzipped_files_dir(year):
    return f"bangdan_data/{year}/"


def unzip_all_bangdan_files():
    """
    将原始微博数据解压缩到当前目录的bangdan_data文件夹
    """
    for year in ANALYSIS_YEARS:
        bangdan_files_dir = get_bangdan_files_dir(year)
        unzipped_dir = get_bangdan_unzipped_files_dir(year)
        extract_7z_files(source_folder=bangdan_files_dir, target_folder=unzipped_dir)


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
            "analyze": BangdanAnalyzer,
        }
    )
