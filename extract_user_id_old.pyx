# distutils: language=c
from libc.string cimport strstr, strlen
from libc.stdlib cimport malloc, free

def extract_user_ids_old(list_of_lines, set_of_user_ids):
    """
    使用 Cython 加速从 JSON 文本中提取 user_id 并匹配
    :param list_of_lines: 输入的文本列表（每行以 \t 分割）
    :param set_of_user_ids: 目标 user_id 集合
    :return: 匹配的文本列表
    """
    cdef list results = []
    cdef list userids = []
    cdef bytes line_bytes, user_id
    cdef const char *c_line, *start, *end
    cdef int col_index, user_id_len
    
    for line in list_of_lines:
        # 将 Python 字符串转换为 bytes
        line_bytes = line.encode('utf-8')  # 持久存储为 bytes 对象
        c_line = line_bytes  # 将 bytes 转换为 C 字符指针
        
        # 按 \t 分割行
        start = c_line
        col_index = 0
        user_id = None
        
        while True:
            # 查找下一个 \t 或行尾
            end = strstr(start, b'\t')
            if not end:
                end = c_line + len(line_bytes)  # 如果没有找到 \t，则到行尾
            
            # 如果是第三列（索引为2），提取 user_id
            if col_index == 4:
                user_id_len = end - start
                user_id = line_bytes[start - c_line : start - c_line + user_id_len]
                break
            
            # 如果到达行尾，退出循环
            if end == c_line + len(line_bytes):
                break
            
            # 更新列索引和起始位置
            col_index += 1
            start = end + 1  # 跳过 \t
        
        # 检查 user_id 是否在目标集合中
        if user_id and user_id in set_of_user_ids:
            results.append(line)
            userids.append(user_id)

    return userids, results