import os
from dotenv import load_dotenv

# ================= 1. 环境与镜像配置 =================
# 强制设置 HF 镜像 (必须在导入 huggingface 库之前设置)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载 .env 文件
# 指向你存放 Key 的绝对路径
ENV_PATH = "/home/wy/Documents/rag/all-in-rag-main/.env"
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    print(f"已加载环境变量: {ENV_PATH}")
else:
    print(f"警告: 未找到环境变量文件: {ENV_PATH}")

# ================= 2. 路径配置 =================
# 项目根目录 (假设本文件在 code/C10)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据路径
DATA_DIR = os.path.join(BASE_DIR, "data", "C10")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CLEAN_MD_DIR = os.path.join(PROCESSED_DIR, "clean_articles")
VECTOR_INDEX_DIR = os.path.join(DATA_DIR, "vector_index")

# ================= 3. Embedding (向量) 模型配置 =================
# 策略: 使用本地 HuggingFace 模型构建向量库 (免费、快速、隐私好)
# 解释: DeepSeek/Moonshot 主要用于生成(Chat)，Embedding 建议用专门的模型。
# 推荐: "BAAI/bge-small-zh-v1.5" (中文效果极佳，体积小)
# 备选: "sentence-transformers/all-MiniLM-L6-v2" (英文为主，你之前的选择)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" 

# ================= 4. LLM (大模型) 配置 (为后续步骤准备) =================
# 这里我们读取你的 DeepSeek 或 Moonshot Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")

# 设置默认使用的 LLM (可以在这里切换 'deepseek' 或 'moonshot')
LLM_PROVIDER = "deepseek" 

if LLM_PROVIDER == "deepseek":
    LLM_API_KEY = DEEPSEEK_API_KEY
    LLM_BASE_URL = "https://api.deepseek.com" # DeepSeek 官方 API 地址
    LLM_MODEL_NAME = "deepseek-chat" # 或 deepseek-reasoner
elif LLM_PROVIDER == "moonshot":
    LLM_API_KEY = MOONSHOT_API_KEY
    LLM_BASE_URL = "https://api.moonshot.cn/v1" # Kimi 官方 API 地址
    LLM_MODEL_NAME = "moonshot-v1-8k"

# 文本切分配置
CHUNK_SIZE = 500 
CHUNK_OVERLAP = 50