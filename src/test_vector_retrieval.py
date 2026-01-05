import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 引入 config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def test_retrieval(query):
    print(f"\nTesting Query: '{query}'")
    
    # 1. 加载索引
    if not os.path.exists(os.path.join(config.VECTOR_INDEX_DIR, "index.faiss")):
        print("错误: 索引文件不存在，请先运行 vector_builder.py")
        return

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        config.VECTOR_INDEX_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True # 本地自己构建的索引是安全的
    )

    # 2. 执行检索
    # k=3 表示返回最相似的 3 个片段
    results = vector_store.similarity_search_with_score(query, k=3)

    # 3. 打印结果
    for i, (doc, score) in enumerate(results):
        print(f"--- Result {i+1} (Score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...") # 只打印前200字符
        print("-" * 30)

if __name__ == "__main__":
    # 测试几个医学问题
    test_retrieval("What is AIVR?")
    test_retrieval("Causes of Sinus Bradycardia")
    test_retrieval("atrial fibrillation stroke risk")