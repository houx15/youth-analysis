import os
import py7zr


def extract_7z_files(source_folder, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for file_name in os.listdir(source_folder):
        # 检查文件是否是.7z文件
        if file_name.endswith(".7z"):
            file_path = os.path.join(source_folder, file_name)
            with py7zr.SevenZipFile(file_path, mode="r") as archive:
                # 解压文件到目标文件夹
                archive.extractall(path=target_folder)
                print(f"Extracted: {file_name}")


def extract_single_7z_file(file_path, target_folder):
    # 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 检查文件是否是.7z文件
    try:
        if file_path.endswith(".7z"):
            with py7zr.SevenZipFile(file_path, mode="r") as archive:
                # 解压文件到目标文件夹
                archive.extractall(path=target_folder)
                print(f"Extracted: {file_path}")
                return "success"
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return None


def sentence_cleaner(sentence: str, src_type: str = "weibo"):
    import re

    if src_type == "tweet":
        sentence = sentence.replace('"', "")
        sentence = sentence.replace("RT", "")
        sentence = sentence.replace(".", "")
        sentence = sentence.replace("'", "")
        results = re.compile(r"[http|https]*://[a-zA-Z0-9.?/&=:_%,-~]*", re.S)
        sentence = re.sub(results, "", sentence)
        sentence = re.sub("[\u4e00-\u9fa5]", "", sentence)
        # results2 = re.compile(r'[@].*?[ ]', re.S)
        # sentence = re.sub(results2, '', sentence)
        sentence = sentence.replace("\n", " ")
        sentence = sentence.strip()
        results2 = re.compile(r"[@].*?[ ]", re.S)
        sentence = re.sub(results2, "", sentence)
        return sentence
    if src_type == "weibo":
        sentence = sentence.replace("“", "")
        sentence = sentence.replace("”", "")
        sentence = sentence.replace("…", "")
        sentence = sentence.replace("点击链接查看更多->", "")
        results = re.compile(
            r"[a-zA-Z0-9.?/&=:_%,-~#《》]", re.S
        )  # 。，：；“”‘’【】（） ]', re.S)
        # results = re.compile(r'[http|https]*://[a-zA-Z0-9.?/&=:_%,-~]*', re.S)
        sentence = re.sub(results, "", sentence)
        results2 = re.compile(r"[//@].*?[:]", re.S)
        sentence = re.sub(results2, "", sentence)
        sentence = sentence.replace("\n", " ")
        sentence = sentence.strip()
        return sentence
    return sentence
