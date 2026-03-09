"""
添加新报纸到省级映射

根据报纸名称判断所属省份，并更新映射文件
"""

import json
import os

MAPPING_FILE = "gender_norms/newspaper_data/newspaper_province_mapping.json"
NAMES_FILE = "gender_norms/newspaper_data/newspaper_names.txt"

# 新报纸到省份的映射（手动判断）
NEW_MAPPINGS = {
    "建筑报": "行业",
    "广西政法报": "广西",
    "河北科技报(农村版)": "河北",
    "中药报": "行业",
    "中国轻工报": "全国",
    "医药导报(中药报)": "行业",
    "内蒙古日报（汉）": "内蒙古",
    "内蒙古日报": "内蒙古",
    "北京电子报": "北京",
    "经济日报农村版": "全国",
    "卫生与生活": "行业",
    "中药事业报": "行业",
    "中国国土资源报(地矿版)": "全国",
    "阿勒泰日报（汉）": "新疆",
    "法治快报(广西政法报)": "广西",
    "大众科技报.科学奥秘周刊": "全国",
    "乌鲁木齐晚报（汉）": "新疆",
    "经济参政报": "全国",
    "日喀则报（汉）": "西藏",
    "格尔木报": "青海",
    "检察日报明镜周刊": "全国",
    "广西法制报": "广西",
    "甘孜日报": "四川",
    "人民日报(海外版)": "全国",
    "光明日报?": "全国",
    "人民日报（海外版）": "全国",
    "海南特区科技报": "海南",
    "-中国机电日报": "全国",
    "四": "未知",
}


def main():
    # 加载已有映射
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    print(f"已有映射: {len(mapping)} 个报纸")
    
    # 读取待映射报纸
    with open(NAMES_FILE, 'r', encoding='utf-8') as f:
        newspapers = [line.strip() for line in f if line.strip()]
    
    print(f"待映射报纸: {len(newspapers)} 个\n")
    
    added = 0
    for newspaper in newspapers:
        if newspaper in mapping:
            print(f"  ⏭️  已存在: {newspaper}")
            continue
        
        if newspaper in NEW_MAPPINGS:
            province = NEW_MAPPINGS[newspaper]
            mapping[newspaper] = {
                "province": province,
                "article_count": 0
            }
            print(f"  ✓ 添加: {newspaper} -> {province}")
            added += 1
        else:
            print(f"  ❓ 未知: {newspaper}")
    
    # 保存
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 完成，新增 {added} 个映射")


if __name__ == "__main__":
    main()
