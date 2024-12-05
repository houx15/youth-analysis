# user_id_counter.pyx

import pandas as pd
from collections import defaultdict
cimport cython

@cython.boundscheck(False)  # 关闭边界检查以提高性能
@cython.wraparound(False)   # 关闭负索引支持以提高性能
def count_user_ids(str file_path):
    # 读取 Parquet 文件
    df = pd.read_parquet(file_path, engine="fastparquet")

    if df.empty:
        return None
    
    # 转换 user_id_binary 到字符串
    user_ids = df['user_id_binary'].apply(lambda x: x.decode('utf-8')).values
    
    # 统计 user_id 出现次数
    user_counts = defaultdict(int)
    cdef int i
    for i in range(len(user_ids)):
        user_counts[user_ids[i]] += 1
    
    return user_counts
