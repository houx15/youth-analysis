# distutils: language=c
from libc.string cimport strstr, strlen
from libc.stdlib cimport malloc, free

def extract_user_ids(list_of_lines, set_of_user_ids):
    """
    使用 Cython 加速从 JSON 文本中提取 user_id 并匹配
    :param list_of_lines: 输入的 JSON 文本列表
    :param set_of_user_ids: 目标 user_id 集合
    :return: 匹配的 JSON 文本列表
    """
    cdef list results = []
    cdef list userids = []
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
                    results.append(line)
                    userids.append(user_id)

    return userids, results