import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 引入配置文件 (需要确保 config.py 在 python 路径中，或者放在同一级目录引用)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_vector_index():
    # 1. 检查目录
    if not os.path.exists(config.CLEAN_MD_DIR):
        print(f"错误: 找不到清洗后的数据目录: {config.CLEAN_MD_DIR}")
        return

    # 2. 加载文档
    print("正在加载 Markdown 文件...")
    documents = []
    md_files = glob.glob(os.path.join(config.CLEAN_MD_DIR, "*.md"))
    
    if not md_files:
        print("未找到 .md 文件，请先运行 Phase 1 的数据清洗脚本。")
        return

    for file_path in md_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            # 将文件名添加到 metadata，方便后续检索时知道来源
            for doc in docs:
                doc.metadata["source_file"] = os.path.basename(file_path)
            documents.extend(docs)
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")

    print(f"共加载 {len(documents)} 个文档。")

    # 3. 文本切分 (Chunking)
    # 医学文本逻辑性强，使用 RecursiveCharacterTextSplitter 尽量保持段落完整
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    splitted_docs = text_splitter.split_documents(documents)
    print(f"文本切分完成，共生成 {len(splitted_docs)} 个文本块 (Chunks)。")

    # 4. 初始化 Embedding 模型
    print(f"正在加载 Embedding 模型: {config.EMBEDDING_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    # 5. 构建 FAISS 索引
    print("正在构建 FAISS 索引 (这可能需要几秒钟)...")
    vector_store = FAISS.from_documents(splitted_docs, embeddings)

    # 6. 保存索引
    if not os.path.exists(config.VECTOR_INDEX_DIR):
        os.makedirs(config.VECTOR_INDEX_DIR)
    
    vector_store.save_local(config.VECTOR_INDEX_DIR)
    print(f"索引构建成功！已保存至: {config.VECTOR_INDEX_DIR}")

if __name__ == "__main__":
    build_vector_index()