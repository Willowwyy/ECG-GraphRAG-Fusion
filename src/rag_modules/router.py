import requests
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def route_query(query):
    """
    判断查询类型:
    - definition: 概念定义，What is... (适合 Vector)
    - relationship: 因果、治疗、区别、并发症 (适合 Graph + Vector)
    """
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""
    Classify the following medical query into one of two categories:
    1. "vector": Simple definition, factual lookup (e.g., "What is AIVR?", "Define AFib").
    2. "hybrid": Complex reasoning, relationships, causes, treatments, comparisons (e.g., "Causes of AIVR", "Difference between A and B", "Drug interactions").
    
    Query: "{query}"
    
    Return ONLY a JSON object: {{"category": "vector"}} or {{"category": "hybrid"}}
    """
    
    payload = {
        "model": config.LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(f"{config.LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=5)
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        data = json.loads(content)
        return data.get("category", "hybrid")
    except Exception as e:
        print(f"Routing failed ({e}), defaulting to hybrid.")
        return "hybrid"