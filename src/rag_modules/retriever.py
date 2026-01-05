import os
import json
from neo4j import GraphDatabase
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class CardioRetriever:
    def __init__(self):
        # 1. 初始化向量库
        print("正在加载 FAISS 向量库...")
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        self.vector_store = FAISS.load_local(
            config.VECTOR_INDEX_DIR, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # 2. 初始化图数据库
        print("正在连接 Neo4j 图数据库...")
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
            auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def search_vector(self, query, k=3):
        """传统向量检索"""
        docs = self.vector_store.similarity_search(query, k=k)
        return "\n\n".join([f"[文档片段]: {d.page_content}" for d in docs])

    def _extract_entities(self, query):
        """利用 LLM 从问题中提取关键实体 (用于图谱定位)"""
        # 简单处理：让 LLM 返回列表
        headers = {
            "Authorization": f"Bearer {config.LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = f"Extract medical entities (diseases, symptoms, drugs) from this query: '{query}'. Return ONLY a JSON list of strings, e.g. [\"Atrial Fibrillation\", \"Stroke\"]. No other text."
        
        payload = {
            "model": config.LLM_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(f"{config.LLM_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=10)
            res_json = response.json()
            content = res_json['choices'][0]['message']['content']
            # 解析 JSON (兼容不同 LLM 可能返回的 key)
            data = json.loads(content)
            if isinstance(data, list): return data
            # 尝试找 values
            for val in data.values():
                if isinstance(val, list): return val
            return []
        except:
            return []

    def search_graph(self, query):
        """图谱检索：实体锚点 -> 邻居遍历"""
        entities = self._extract_entities(query)
        print(f"   (图谱) 提取到的实体: {entities}")
        
        if not entities:
            return "No specific entities found in knowledge graph."

        context_list = []
        with self.driver.session() as session:
            for entity in entities:
                # 模糊匹配节点 ID
                cypher = """
                MATCH (n)-[r]-(m)
                WHERE toLower(n.id) CONTAINS toLower($entity)
                RETURN n.id, type(r), m.id, m.label
                LIMIT 15
                """
                result = session.run(cypher, entity=entity)
                
                found = False
                for record in result:
                    found = True
                    # 格式化为自然语言: "AIVR RELATED_TO Ventricular Tachycardia"
                    rel_text = f"{record['n.id']} --[{record['type(r)']}]--> {record['m.id']} ({record.get('m.label', 'Unknown')})"
                    context_list.append(rel_text)
                
                if not found:
                    # 尝试只查节点定义
                    cypher_node = "MATCH (n) WHERE toLower(n.id) CONTAINS toLower($entity) RETURN n.id, labels(n) LIMIT 1"
                    res_node = session.run(cypher_node, entity=entity)
                    for rec in res_node:
                        context_list.append(f"Found Entity: {rec['n.id']} (Type: {rec['labels(n)']})")

        return "\n".join(context_list) if context_list else "No relevant graph relationships found."

    def hybrid_search(self, query, mode="hybrid"):
        """混合检索入口"""
        vector_res = ""
        graph_res = ""
        
        if mode in ["vector", "hybrid"]:
            vector_res = self.search_vector(query)
            
        if mode in ["graph", "hybrid"]:
            try:
                graph_res = self.search_graph(query)
            except Exception as e:
                print(f"Graph search failed: {e}")
                graph_res = ""
        
        return {
            "vector_context": vector_res,
            "graph_context": graph_res
        }