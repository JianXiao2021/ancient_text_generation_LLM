import os
import json
import random
from config.config import DATA_PATH

def extract_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):  # 每三行一个组
            if i + 1 < len(lines):  # 防止越界
                ancient_text = lines[i].strip().replace("古文：", "")
                modern_text = lines[i+1].strip().replace("现代文：", "")
                pair = {
                    "input": f"现代文：{modern_text} 古文：",
                    "output": ancient_text
                }
                pairs.append(pair)
    return pairs

def recursive_search_and_extract(root_dir):
    all_pairs = []
    for root, dirs, files in os.walk(root_dir):
        if "bitext.txt" in files:
            file_path = os.path.join(root, "bitext.txt")
            pairs = extract_pairs(file_path)
            all_pairs.extend(pairs)
    return all_pairs

def split_data(pairs, test_ratio=0.2):
    random.shuffle(pairs)
    test_size = int(len(pairs) * test_ratio)
    test_set = pairs[:test_size]
    train_set = pairs[test_size:]
    return train_set, test_set

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    root_dir = os.path.join(DATA_PATH, "original")
    output_dir = os.path.join(DATA_PATH, "processed")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取所有bitext.txt文件中的内容
    all_pairs = recursive_search_and_extract(root_dir)
    
    # 分割数据为训练集、验证集和测试集
    train_set, test_set = split_data(all_pairs, test_ratio=0.2)
    test_set, valid_set = split_data(test_set, test_ratio=0.5)
    
    # 保存数据
    save_json(train_set, os.path.join(output_dir, "train_data.json"))
    save_json(valid_set, os.path.join(output_dir, "validation_data.json"))
    save_json(test_set, os.path.join(output_dir, "test_data.json"))

    print(f"数据处理完成：训练集 {len(train_set)} 条，验证集 {len(valid_set)} 条，测试集 {len(test_set)} 条。")
