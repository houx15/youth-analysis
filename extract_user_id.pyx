# distutils: language=c
from libc.string cimport strstr, strlen
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

def extract_user_ids_optimized(list_of_lines, set_of_user_ids):
    """
    使用 Cython 加速从 JSON 文本中提取 user_id 并匹配
    使用 NumPy 数组预分配存储结果，避免 append 操作
    :param list_of_lines: 输入的 JSON 文本列表
    :param set_of_user_ids: 目标 user_id 集合
    :return: 匹配的 JSON 文本列表和 user_id 列表
    """
    cdef int num_lines = len(list_of_lines)
    cdef np.ndarray results = np.empty(num_lines, dtype=object)  # 用于存储匹配的 JSON 文本
    cdef np.ndarray userids = np.empty(num_lines, dtype=object)  # 用于存储匹配的 user_id
    cdef int result_idx = 0  # 当前存储索引
    cdef bytes line_bytes, user_id
    cdef const char *c_line, *start, *end
    cdef int key_len, user_id_len
    
    # 初始化 key_len
    key_len = len('"user_id":"')

    for line in list_of_lines:
        # 将 Python 字符串转换为 bytes
        line_bytes = line.encode('utf-8')  # 持久存储为 bytes 对象
        c_line = line_bytes  # 将 bytes 转换为 C 字符指针
        
        # 查找 "user_id":" 的起始位置
        start = strstr(c_line, b'"user_id":"')
        if start:
            start += key_len  # 跳过 "user_id":" 的长度
            end = strstr(start, b'"')  # 查找结束引号
            if end:
                # 提取 user_id（直接使用 Python 的 bytes 对象，无需 malloc）
                user_id_len = end - start
                user_id = line_bytes[start - c_line : start - c_line + user_id_len]
                
                # 检查 user_id 是否在目标集合中
                if user_id in set_of_user_ids:
                    results[result_idx] = line  # 存储匹配的 JSON 文本
                    userids[result_idx] = user_id  # 存储匹配的 user_id
                    result_idx += 1

    # 使用 NumPy 的切片操作去掉多余的空值
    return userids[:result_idx].tolist(), results[:result_idx].tolist()