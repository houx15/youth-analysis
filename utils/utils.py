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