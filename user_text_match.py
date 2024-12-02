import os
import json
import time
from datetime import datetime, timedelta

from configs.configs import *
from utils.utils import *

TASK_ID = 3
OUTPUT_DIR = "match_working_data"

task_id_to_date = {
    0: datetime(2020, 1, 28),
    1: datetime(2020, 2, 12), # c
    2: datetime(2020, 3, 1), # c
    3: datetime(2020, 3, 17),
    4: datetime(2020, 4, 10), # c
    5: datetime(2020, 5, 1), # c
    6: datetime(2020, 5, 20),
    7: datetime(2020, 6, 15), # c
    8: datetime(2020, 7, 1), # c
    9: datetime(2020, 8, 27),
    10: datetime(2020, 9, 7), # c
}

task_id_to_date = {
    0: datetime(2020, 2, 27),
    1: datetime(2020, 3, 1), # c
    2: datetime(2020, 2, 27),
    3: datetime(2020, 3, 1), # c
    4: datetime(2020, 3, 31),
    5: datetime(2020, 4, 10), # c
    6: datetime(2020, 4, 24),
    7: datetime(2020, 5, 1), # c
    8: datetime(2020, 7, 1), # c
    9: datetime(2020, 8, 27),
    10: datetime(2020, 9, 7), # c
}


complement_set = [datetime(2020, 2, 11), datetime(2020, 2, 29), datetime(2020, 4, 9), datetime(2020, 4, 30), datetime(2020, 6, 14), datetime(2020, 6, 29), datetime(2020, 6, 30), datetime(2020, 9, 6)]

with open(os.path.join(OUTPUT_DIR, "user_to_text.json"), "r") as f:
    date_user_id_to_text_id = json.load(f)

with open(os.path.join(OUTPUT_DIR, "date_covered_users.json"), "r") as f:
    date_covered_users_serializable = json.load(f)


def get_zipped_fresh_data_file(year, date):
    """
    date should be yyyy-mm-dd format
    """
    return f"{ORIGIN_DATA_DIR}/{year}/freshdata/weibo_freshdata.{date}.7z"


def get_unzipped_fresh_data_folder(year):
    return f"text_working_data/{year}/"


def get_unzipped_fresh_data_file(year, date):
    return f"text_working_data/{year}/weibo_freshdata.{date}"


def delete_unzipped_fresh_data_file(year, date):
    """
    处理完毕之后需要删除文件
    """
    file_path = get_unzipped_fresh_data_file(year, date)
    # 删除文件
    try:
        os.remove(file_path)
        print(f"文件 {file_path} 已成功删除。")
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在。")
    except Exception as e:
        print(f"删除文件时发生错误: {e}")


def log_id(user_id, original_id):
    with open(f"user_id_to_original_id_{TASK_ID}.txt", "a") as f:
        f.write(f"{user_id},{original_id}\n")


def unzip_one_fresh_data_file(year, date):
    """
    date should be yyyy-mm-dd format
    """
    unzipped_file_path = get_unzipped_fresh_data_file(year, date)
    
    # 检查文件是否已经解压
    if os.path.exists(unzipped_file_path):
        print(f"文件 {unzipped_file_path} 已经存在，跳过解压。")
        return unzipped_file_path

    # 如果文件不存在，则进行解压
    zipped_file_path = get_zipped_fresh_data_file(year, date)
    unzipped_dir = get_unzipped_fresh_data_folder(year)
    result = extract_single_7z_file(file_path=zipped_file_path, target_folder=unzipped_dir)
    
    if result == "success":
        return unzipped_file_path
    return None


with open("data/weibo_cov_user_to_original_id.json", "r") as f:
    weibo_cov_user_to_original_id = json.load(f)


# 改成从task_id选取事件 id-1为开始时间，id前一天为结束时间。从dict中拿到的是YYYY-mm-dd需要转唯datetime
start_date = task_id_to_date[TASK_ID-1]
end_date = task_id_to_date[TASK_ID] - timedelta(days=1)
current_date = start_date

for current_date in (start_date + timedelta(days=n) for n in range((end_date - start_date).days + 1)):
    start = int(time.time())
    date = current_date.strftime("%Y-%m-%d")
    user_to_text = date_user_id_to_text_id.get(date)
    if user_to_text is None:
        continue
    date_effective_users = date_covered_users_serializable[date]

    # 获取date-1日的数据， date是yyyy-mm-dd格式 

    previous_date = current_date - timedelta(days=1)
    previous_user_to_text = date_user_id_to_text_id.get(previous_date.strftime("%Y-%m-%d"), {})
    user_to_text.update(previous_user_to_text)
    del previous_user_to_text

    previous_date_effective_users = date_covered_users_serializable.get(previous_date.strftime("%Y-%m-%d"), [])
    # 更新到date_effective_users中，date_effective_users是list
    date_effective_users.extend(previous_date_effective_users)
    del previous_date_effective_users

    next_date = current_date + timedelta(days=1)
    next_user_to_text = date_user_id_to_text_id.get(next_date.strftime("%Y-%m-%d"), {})
    user_to_text.update(next_user_to_text)
    del next_user_to_text

    next_date_effective_users = date_covered_users_serializable.get(next_date.strftime("%Y-%m-%d"), [])
    date_effective_users.extend(next_date_effective_users)
    del next_date_effective_users


    # date_text_to_user = {v: k for k, v in user_to_text.items() if (k in date_effective_users and k not in weibo_cov_user_to_original_id.keys())}
    date_text_to_user = {v: k for k, v in user_to_text.items() if k in date_effective_users}
    date_text_id_set = set(date_text_to_user.keys())
    if len(date_text_id_set) == 0:
        print(f"date {date} has no text to match")
        continue

    file_path = unzip_one_fresh_data_file(2020, date)

    print(f"unzipped one file in {int(time.time()) - start} seconds")
    
    try:
        if date == "2020-06-30":
            file_path = f"text_working_data/2020/weibo_2020-06-30.csv"
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                for line in file:
                    """
                    "46890032291","2020-06-30 00:12:36","1593447156000","1","2789934082","妞子蓝楸瑛","https://tva1.sinaimg.cn/crop.0.0.180.180.50/a64b0402jw1e8qgp5bmzyj2050050aa8.jpg?KID=imgbed,tva&Expires=1593457954&ssig=WuAFhSJ49R","普通用户","4520977143508623","转发微博","0","0","0","J8NI6eIKH","微博 weibo.com","","2020-06-29 02:20:09","1593368409","2920534890","地盘鲁路修兰佩洛基1986","普通用户","4247252011050572","双子座 今日(6月4日)综合运势：5，幸运颜色：粉色，幸运数字：7，速配星座：天蝎座（分享自@微心情） 查看更多：http://t.cn/h5gw6 ​​​","95","0","0","GjPeV4piI","微博 weibo.com","","2018-06-04 18:14:11","1528107251","","0","","0","0","0","0","2020-06-30"
                    """
                    line_data = line.strip().split(",")
                    try:
                        if line_data[13] in date_text_id_set:
                            # weibo_cov_user_to_original_id[date_text_to_user[line_data[13]]] = line_data[4]
                            log_id(date_text_to_user[line_data[13]], line_data[4])
                            date_text_id_set.remove(line_data[13])
                            if len(date_text_id_set) == 0:
                                break
                    except IndexError as e:
                        print(f"IndexError occurred: {e}")
                        continue
        else:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                for line in file:
                    """
                    40984940671        {"id":"40984940671","crawler_time":"2020-01-01 04:27:59","crawler_time_stamp":"1577824079000","is_retweet":"0","user_id":"5706021763","nick_name":"诗词歌赋","tou_xiang":"https:\/\/tvax2.sinaimg.cn\/crop.0.0.1002.1002.50\/006e9SV5ly8g4yg7ozexlj30ru0ruabp.jpg?KID=imgbed,tva&Expires=1577834878&ssig=lHvYHGBxwq","user_type":"黄V","weibo_id":"4455589780114474","weibo_content":"给自己设立一个目标，给自己未来一个明确的希望，给自己的生活一个方向灯。冲着这个方向而努力，不断去超越自己，提高自己的水平，不能让自己有懈怠的时候。早安! ","zhuan":"0","ping":"0","zhan":"0","url":"Ink8W0tMm","device":"Redmi Note 7 Pro","locate":"","time":"2019-12-31 15:54:07","time_stamp":"1577778847","r_user_id":"","r_nick_name":"","r_user_type":"","r_weibo_id":"","r_weibo_content":"","r_zhuan":"","r_ping":"","r_zhan":"","r_url":"","r_device":"","r_location":"","r_time":"","r_time_stamp":"","pic_content":"","src":"4","tag":"106750860151","vedio":"0","vedio_image":"","edited":"0","r_edited":"","isLongText":"0","r_isLongText":"","lat":"","lon":"","d":"2020-01-01"}
                    """
                    line_data = line.strip().split("\t")
                    try:
                        data = json.loads(line_data[1])
                    except IndexError as e:
                        print(f"IndexError occurred: {e}")
                        continue
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: {e}")
                        # 打印出错误位置
                        print(f"Error at line {e.lineno}, column {e.colno}")
                        # 打印出错误字符位置
                        print(f"Error at character {e.pos}, {line_data[1][int(e.pos)-20: int(e.pos)+20]}")
                        continue
                    
                    if data['url'] in date_text_id_set:
                        # weibo_cov_user_to_original_id[date_text_to_user[data['url']]] = data['user_id']
                        log_id(date_text_to_user[data['url']], data['user_id'])
                        date_text_id_set.remove(data['url'])
                        if len(date_text_id_set) == 0:
                            break
                    
                    # 如果data time的日期部分已经大于previous date了，就break
                    # if data['time'][:10] > previous_date.strftime("%Y-%m-%d"):
                    #     break
        
        print(f"processed one file in {int(time.time()) - start} seconds")
    except:
        print(f"Error occurred when processing file {file_path}")
        # current_date += timedelta(days=1)
        continue

    delete_unzipped_fresh_data_file(2020, date)

    # current_date += timedelta(days=1)


# with open(f"data/weibo_cov_user_to_original_id_{TASK_ID}.json", "w") as f:
#     json.dump(weibo_cov_user_to_original_id, f)

