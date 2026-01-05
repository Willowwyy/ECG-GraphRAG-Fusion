import os
import json
import time
import glob
from typing import List, Dict
import requests
from tqdm import tqdm # 进度条库，如果没有请 pip install tqdm

# 引入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ================= 定义图谱 Schema (本体) =================
# 限制 LLM 只能提取这些类型的节点和关系，防止图谱过于杂乱
NODE_TYPES = [
    "Disease",       # 疾病 (如 Atrial Fibrillation)
    "Symptom",       # 症状/体征 (如 Chest pain, Palpitations)
    "ECG_Feature",   # 心电特征 (如 ST elevation, Delta wave)
    "Medication",    # 药物 (如 Amiodarone, Aspirin)
    "Anatomy",       # 解剖部位 (如 Left Ventricle, AV Node)
    "Treatment"      # 治疗手段 (如 Cardioversion, PCI)
]

RELATION_TYPES = [
    "HAS_SYMPTOM",   # 疾病 -> 症状
    "SHOWS_ON_ECG",  # 疾病 -> 心电特征
    "TREATED_WITH",  # 疾病 -> 药物/治疗
    "CAUSES",        # 疾病 -> 疾病 / 病因 -> 疾病
    "AFFECTS",       # 疾病 -> 解剖部位
    "CONTRAINDICATES"# 禁忌 (如 WPW -> Digoxin)
]

def call_llm_extraction(text, filename):
    """
    调用 DeepSeek API 进行信息抽取
    """
    
    prompt = f"""
    You are an expert cardiologist data analyst. Your task is to extract a structured Knowledge Graph from the provided medical text.
    
    Target Node Types: {", ".join(NODE_TYPES)}
    Target Relationship Types: {", ".join(RELATION_TYPES)}

    Text Source: {filename}
    Text Content:
    {text[:4000]}  # 限制长度防止 Token 溢出，对于长文通常只取核心部分或分段处理

    Output Requirement:
    Return ONLY a valid JSON object with the following structure. Do not add any markdown formatting or explanation.
    {{
        "nodes": [
            {{"id": "Exact Name", "label": "Type", "source": "{filename}"}}
        ],
        "relationships": [
            {{"source": "Source Node Id", "target": "Target Node Id", "type": "RELATION_TYPE", "description": "Short context"}}
        ]
    }}
    """

    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config.LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs strict JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1, # 低温度保证输出稳定
        "response_format": {"type": "json_object"} # 强制 JSON 模式 (如果 API 支持)
    }

    try:
        response = requests.post(f"{config.LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        print(f"\n[Error] Failed to extract from {filename}: {e}")
        return None

def build_knowledge_graph_data():
    # 1. 准备路径
    if not os.path.exists(config.CLEAN_MD_DIR):
        print("Error: Cleaned data directory not found.")
        return
    
    output_file = os.path.join(config.PROCESSED_DIR, "import_graph_data.json")
    
    md_files = glob.glob(os.path.join(config.CLEAN_MD_DIR, "*.md"))
    print(f"Found {len(md_files)} articles to process.")

    all_nodes = {} # 使用字典去重: ID -> NodeObj
    all_relations = []

    # 2. 遍历文件进行提取
    for file_path in tqdm(md_files, desc="Extracting Knowledge"):
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 调用 LLM
        data = call_llm_extraction(text, filename)
        
        if data:
            # 处理节点 (去重)
            for node in data.get("nodes", []):
                node_id = node.get("id")
                if node_id:
                    # 简单的归一化：转小写首字母大写，去除空格
                    clean_id = node_id.strip()
                    if clean_id not in all_nodes:
                        all_nodes[clean_id] = {
                            "id": clean_id,
                            "label": node.get("label", "Concept"),
                            "source": filename
                        }
            
            # 处理关系
            for rel in data.get("relationships", []):
                src = rel.get("source")
                tgt = rel.get("target")
                if src and tgt:
                    all_relations.append({
                        "source": src.strip(),
                        "target": tgt.strip(),
                        "type": rel.get("type", "RELATED_TO"),
                        "source_file": filename
                    })
        
        # 避免 API 速率限制 (Rate Limit)，稍微 sleep 一下
        time.sleep(1)

    # 3. 保存结果
    final_data = {
        "nodes": list(all_nodes.values()),
        "relationships": all_relations
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nExtraction complete!")
    print(f"Nodes: {len(final_data['nodes'])}")
    print(f"Relationships: {len(final_data['relationships'])}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    # 安装 tqdm 用于显示进度条: pip install tqdm
    build_knowledge_graph_data()