import os
from rag_modules.retriever import CardioRetriever
from rag_modules.router import route_query
import config
import requests

def generate_answer(query, contexts):
    """最终调用 LLM 生成回答"""
    # 组装 Prompt
    final_prompt = f"""
    You are an expert Cardiologist Assistant. Answer the user query based ONLY on the provided Context.
    
    --- Context from Medical Guidelines (Text) ---
    {contexts['vector_context']}
    
    --- Context from Knowledge Graph (Relationships) ---
    {contexts['graph_context']}
    
    --- User Query ---
    {query}
    
    --- Instruction ---
    Synthesize the information. If the Knowledge Graph provides specific causes or relations, highlight them.
    If the answer is not in the context, state "I don't have enough information".
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
    
    print("\nThinking...")
    response = requests.post(f"{config.LLM_BASE_URL}/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

def main():
    # 1. 初始化
    retriever = CardioRetriever()
    
    # 2. 测试查询
    # test_query = "What is AIVR?"  # 简单定义
    test_query = "What causes AIVR and how is it related to MI?" # 复杂关系
    
    print(f"\n====== User Query: {test_query} ======")
    
    # 3. 路由
    route = route_query(test_query)
    print(f"🤖 Router Decision: {route.upper()} Search")
    
    # 4. 检索
    contexts = retriever.hybrid_search(test_query, mode=route)
    
    print("-" * 20 + " Retrieved Vector Context (Snippet) " + "-" * 20)
    print(contexts['vector_context'][:300] + "...\n")
    
    print("-" * 20 + " Retrieved Graph Context (Snippet) " + "-" * 20)
    print(contexts['graph_context'][:500] + "...\n")
    
    # 5. 生成
    answer = generate_answer(test_query, contexts)
    print("\n" + "="*20 + " FINAL ANSWER " + "="*20)
    print(answer)
    
    retriever.close()

if __name__ == "__main__":
    main()