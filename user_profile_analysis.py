from configs.configs import *
from utils.utils import extract_7z_files


def get_user_profile_files_dir(year):
    return f"{ORIGIN_DATA_DIR}/{year}/user_profile/"


def get_user_profile_unzipped_files_dir(year):
    return f"user_profile/{year}/"


def unzip_year_huati_bang_files(year):
    """
    将原始微博数据解压缩到当前目录的bangdan_data文件夹
    """
    bangdan_files_dir = get_user_profile_files_dir(year)
    unzipped_dir = get_user_profile_unzipped_files_dir(year)
    extract_7z_files(source_folder=bangdan_files_dir, target_folder=unzipped_dir)


if __name__ == "__main__":
    unzip_year_huati_bang_files(2023)