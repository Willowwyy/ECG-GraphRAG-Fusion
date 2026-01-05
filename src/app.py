import streamlit as st
import os
import sys

# 将项目路径加入系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_modules.retriever import CardioRetriever
from rag_modules.router import route_query
from rag_modules.generation import generate_answer  
import config

# --- 页面配置 ---
st.set_page_config(page_title="CardioGraphRAG", page_icon="🫀", layout="wide")

st.title("🫀 CardioGraphRAG: 智能心电助手")
st.markdown(f"**Engine**: GraphRAG (Neo4j) + VectorRAG (FAISS) + {config.LLM_MODEL_NAME}")

# --- 初始化 Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = CardioRetriever()

# --- 侧边栏 ---
with st.sidebar:
    st.header("🔍 系统状态")
    st.success("✅ FAISS 向量库已连接")
    st.success("✅ Neo4j 图谱已连接")
    
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

# --- 核心逻辑 ---
def get_bot_response(user_query):
    retriever = st.session_state.retriever
    
    # 1. 路由与检索 UI 展示
    with st.status("正在思考...", expanded=True) as status:
        st.write("🤔 分析意图...")
        route = route_query(user_query)
        st.write(f"👉 决策: **{route.upper()}** 模式")
        
        st.write("🔍 检索知识库...")
        contexts = retriever.hybrid_search(user_query, mode=route)
        
        # 如果有图谱结果，展示给用户看 (增加可解释性)
        if contexts['graph_context']:
            with st.expander("查看图谱推理路径 (Knowledge Graph)"):
                st.code(contexts['graph_context'], language="text")
        
        status.update(label="检索完成! 正在生成回答...", state="complete", expanded=False)
    
    # 2. 生成回答 (调用 generation.py)
    return generate_answer(user_query, contexts)

# --- 聊天界面渲染 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 处理输入 ---
if prompt := st.chat_input("请输入问题 (例如: What causes AIVR?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_bot_response(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})