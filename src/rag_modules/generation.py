import requests
import json
import os
import sys

# 引入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_answer(query, contexts):
    """
    封装调用 LLM 生成回答的逻辑
    """
    # 组装 Prompt
    final_prompt = f"""
    You are an expert Cardiologist Assistant. Answer the user query based ONLY on the provided Context.
    
    --- Context from Medical Guidelines (Text) ---
    {contexts.get('vector_context', '')}
    
    --- Context from Knowledge Graph (Relationships) ---
    {contexts.get('graph_context', '')}
    
    --- User Query ---
    {query}
    
    --- Instruction ---
    Synthesize the information. If the Knowledge Graph provides specific causes or relations, highlight them.
    Be professional, concise, and structured.
    """
    
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": final_prompt}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(f"{config.LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
        res_json = response.json()
        return res_json['choices'][0]['message']['content']
    except Exception as e:
        return f"Error generating answer: {str(e)}"