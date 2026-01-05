import os
import json
import time
from neo4j import GraphDatabase
from tqdm import tqdm

# 引入配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """清空数据库 (慎用)"""
        print("正在清空 Neo4j 数据库...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("数据库已清空。")

    def create_constraints(self):
        """创建唯一性约束，加速查询并防止重复"""
        print("正在创建索引和约束...")
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:ECG_Feature) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medication) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Anatomy) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Treatment) REQUIRE t.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE" # 兜底类型
        ]
        with self.driver.session() as session:
            for query in constraints:
                session.run(query)

    def import_data(self, json_path):
        if not os.path.exists(json_path):
            print(f"错误: 找不到数据文件 {json_path}")
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        relationships = data.get("relationships", [])

        print(f"准备导入 {len(nodes)} 个节点和 {len(relationships)} 条关系...")

        with self.driver.session() as session:
            # 1. 批量创建节点
            print("正在导入节点...")
            batch_size = 100
            for i in tqdm(range(0, len(nodes), batch_size)):
                batch = nodes[i:i + batch_size]
                # 使用 MERGE 避免重复
                query = """
                UNWIND $batch AS row
                CALL {
                    WITH row
                    MERGE (n:Concept {id: row.id})
                    SET n.name = row.id,
                        n.source = row.source
                    # 动态设置标签 (Label)
                    WITH n, row
                    CALL apoc.create.addLabels(n, [row.label]) YIELD node
                    RETURN count(*)
                } IN TRANSACTIONS
                RETURN count(*)
                """
                # 注意：如果服务器没有安装 APOC 插件，上面的动态 Label 会报错。
                # 兼容方案：简单地按类型写入。为了稳妥，我们用纯 Cypher 方案：
                
                # ------ 纯 Cypher 方案 (更通用) ------
                for node in batch:
                    label = node.get("label", "Concept").replace(" ", "_") # 防止非法字符
                    cypher = f"MERGE (n:`{label}` {{id: $id}}) SET n.source = $source"
                    session.run(cypher, id=node['id'], source=node.get('source', ''))

            # 2. 批量创建关系
            print("正在导入关系...")
            for i in tqdm(range(0, len(relationships), batch_size)):
                batch = relationships[i:i + batch_size]
                for rel in batch:
                    rel_type = rel.get("type", "RELATED_TO").upper().replace(" ", "_")
                    # 查找起点和终点，建立关系
                    # 注意：这里假设节点已经通过 ID 建立好了
                    cypher = f"""
                    MATCH (source) WHERE source.id = $source_id
                    MATCH (target) WHERE target.id = $target_id
                    MERGE (source)-[r:`{rel_type}`]->(target)
                    SET r.source_file = $file
                    """
                    session.run(cypher, 
                                source_id=rel['source'], 
                                target_id=rel['target'], 
                                file=rel.get('source_file', ''))

        print("图谱导入完成！")

if __name__ == "__main__":
    # 读取 .env 中的配置
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    importer = Neo4jImporter(neo4j_uri, neo4j_user, neo4j_password)
    
    # 路径
    json_file = os.path.join(config.PROCESSED_DIR, "import_graph_data.json")
    
    try:
        importer.clear_database() # ⚠️ 测试阶段建议先清空，生产环境请注释掉
        importer.create_constraints()
        importer.import_data(json_file)
    except Exception as e:
        print(f"导入出错: {e}")
        print("提示: 确保 Neo4j 正在运行，且 .env 配置正确。")
    finally:
        importer.close()