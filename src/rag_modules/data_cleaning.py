import os
import re
import ast
import pandas as pd
import json

# ================= 配置路径 =================
# 请根据你的实际路径调整
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW_DIR = os.path.join(BASE_DIR, "data", "C10", "raw")
LITFL_DIR = os.path.join(RAW_DIR, "litfl_articles")
CSV_PATH = os.path.join(RAW_DIR, "ptbxl_database.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "C10", "processed")
CLEAN_MD_DIR = os.path.join(PROCESSED_DIR, "clean_articles")

def clean_litfl_markdown(text):
    """
    清洗 LITFL 的 Markdown 文本，移除页眉、作者、页脚、引用等无关信息。
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # 标记是否在正文区域
    is_content = True
    
    for line in lines:
        stripped_line = line.strip()
        
        # 1. 跳过作者行 (通常在标题下，以 - 开头，后面跟人名)
        if stripped_line.startswith("- Chris Nickson") or stripped_line.startswith("- Mike Cadogan"):
            continue
            
        # 2. 识别结束标志 (遇到这些关键词通常意味着正文结束)
        # LITFL 文章底部通常有 'LITFL', 'Journal articles', 'References', 'Critical Care' (Bio)
        end_markers = ["## Critical Care", "Journal articles", "References", "LITFL"]
        if any(marker in stripped_line for marker in end_markers):
            is_content = False
            
        # 3. 如果不在内容区，停止处理
        if not is_content:
            break
            
        # 4. 去除多余空行，保留非空行
        if stripped_line:
            cleaned_lines.append(line)
            
    return '\n'.join(cleaned_lines)

def process_markdown_files():
    """读取 raw 目录下的 md 文件，清洗后存入 processed 目录"""
    if not os.path.exists(LITFL_DIR):
        print(f"警告: 文件夹 {LITFL_DIR} 不存在，跳过 Markdown 处理。")
        return

    print(f"正在处理 Markdown 文件: {LITFL_DIR} ...")
    count = 0
    for filename in os.listdir(LITFL_DIR):
        if filename.endswith(".md"):
            file_path = os.path.join(LITFL_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                cleaned_content = clean_litfl_markdown(content)
                
                # 保存清洗后的文件
                save_path = os.path.join(CLEAN_MD_DIR, filename)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                count += 1
            except Exception as e:
                print(f"处理文件 {filename} 出错: {e}")
    
    print(f"成功清洗并保存了 {count} 个 Markdown 文件到 {CLEAN_MD_DIR}")

def process_ptbxl_csv():
    """
    从 ptbxl_database.csv 中提取诊断代码 (scp_codes)
    作为构建知识图谱的'疾病实体'基础。
    """
    if not os.path.exists(CSV_PATH):
        print(f"警告: 文件 {CSV_PATH} 不存在，跳过 CSV 处理。")
        return

    print(f"正在处理 CSV 文件: {CSV_PATH} ...")
    try:
        df = pd.read_csv(CSV_PATH)
        
        # scp_codes 列通常是字符串格式的字典: "{'NORM': 100.0, 'LBBB': 0.0}"
        # 我们需要解析它并提取 Key (如 NORM, LBBB)
        unique_conditions = set()
        
        if 'scp_codes' in df.columns:
            for val in df['scp_codes'].dropna():
                try:
                    # 将字符串转为字典
                    code_dict = ast.literal_eval(val)
                    # 提取 keys (疾病代码)
                    for code in code_dict.keys():
                        unique_conditions.add(code)
                except:
                    continue
        
        # 保存为 JSON
        entities_path = os.path.join(PROCESSED_DIR, "entities.json")
        data_to_save = {
            "source": "PTB-XL",
            "type": "Disease_Codes",
            "count": len(unique_conditions),
            "entities": list(unique_conditions)
        }
        
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
            
        print(f"提取了 {len(unique_conditions)} 个唯一的诊断代码，已保存至 {entities_path}")

    except Exception as e:
        print(f"处理 CSV 出错: {e}")

if __name__ == "__main__":
    # 确保输出目录存在
    if not os.path.exists(CLEAN_MD_DIR):
        os.makedirs(CLEAN_MD_DIR)
        
    process_markdown_files()
    process_ptbxl_csv()