# deduplicate.pyx

from cython cimport boundscheck, wraparound
from libc.string cimport strstr

@boundscheck(False)
@wraparound(False)
def deduplicate_by_weibo_id(list_of_lines):
    cdef set seen_weibo_ids = set()
    cdef int n = len(list_of_lines)
    cdef int i
    cdef str weibo_id
    is_unique = [1] * n  # 使用 Python 列表

    for i in range(n):
        # 提取 weibo_id
        weibo_id = extract_weibo_id(list_of_lines[i])
        
        if weibo_id == "":
            continue  # 跳过空字符串
        
        if weibo_id in seen_weibo_ids:
            is_unique[i] = 0  # 标记为重复
        else:
            seen_weibo_ids.add(weibo_id)

    return is_unique

cdef str extract_weibo_id(bytes line):
    cdef const char *c_line
    cdef const char *start
    cdef const char *end
    cdef int key_len = len(b'"weibo_id":"')
    
    # 将 Python 字节串转换为 C 字符数组
    c_line = <const char *>line
    
    start = strstr(c_line, b'"weibo_id":"')
    if start:
        start += key_len
        end = strstr(start, b'"')
        if end:
            return line[start - c_line : start - c_line + (end - start)].decode('utf-8')

    return ""  # 未找到时返回空字符串
