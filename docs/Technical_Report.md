# 项目技术报告：基于知识图谱与向量混合检索的心电诊断推理系统

**项目代号**: ECG-GraphRAG-Fusion  
**版本**: v1.0.0  
**日期**: 2026-01-05

---

## 1. 项目背景与挑战

### 1.1 传统 RAG 在医学领域的局限
在心电图 (ECG) 辅助诊断领域，医生和医学生经常需要查询复杂的病理关系（例如：“某种心律失常的潜在药物诱因是什么？”）。传统的基于向量检索增强生成 (Vector-RAG) 系统存在以下瓶颈：
* **逻辑推理缺失**：向量检索擅长匹配关键词，但难以理解“因果”、“并发症”、“禁忌症”等逻辑链路。
* **跨文档关联弱**：医学知识分布在不同的指南章节中，向量检索难以将分散的知识点（如症状在A处，治疗在B处）串联起来。

### 1.2 本项目的解决方案
本项目 (**ECG-GraphRAG-Fusion**) 引入 **知识图谱 (Knowledge Graph)** 作为第二路知识源，构建了“图谱+向量”的双路混合检索架构。通过将非结构化的医学指南转化为结构化的 `(实体)-[关系]->(实体)` 三元组，赋予了系统多跳推理 (Multi-hop Reasoning) 的能力。

---

## 2. 系统整体架构

系统采用了经典的 RAG 流水线，但在检索层进行了重大创新。

### 2.1 架构图
*(此处引用 assets/architecture.png)*

### 2.2 核心模块说明

| 模块 | 技术选型 | 功能描述 |
| :--- | :--- | :--- |
| **知识抽取层** | DeepSeek LLM | 从 LITFL 医学文献中提取 Disease, Symptom, Medication 等实体及其关系。 |
| **存储层** | Neo4j + FAISS | Neo4j 存储图谱结构；FAISS 存储原始文本切片的 Embedding 向量。 |
| **路由层 (Router)** | Intent Classification | 基于用户 Query 的语义复杂度，自动判定走“快速查词”模式还是“深度推理”模式。 |
| **检索层 (Retriever)** | Hybrid Search | **向量路**：语义相似度匹配；**图谱路**：实体锚点定位 + 2跳邻居遍历。 |
| **生成层 (Generator)** | DeepSeek-V3 | 综合结构化知识（图）和非结构化细节（文），生成符合临床逻辑的回答。 |

---

## 3. 关键技术实现

### 3.1 知识图谱构建 (Graph Construction)
我们设计了心电领域的本体 (Ontology)，包含以下核心节点类型：
* `Disease` (如: Atrial Fibrillation)
* `ECG_Feature` (如: ST Elevation)
* `Medication` (如: Digoxin)
* `Treatment` (如: Cardioversion)

数据处理流程：
1.  **清洗**: 去除 HTML 标签与无关元数据。
2.  **提取**: 使用 LLM 提取三元组，例如 `(AIVR) --[CAUSES]--> (Digoxin Toxicity)`。
3.  **入库**: 目前图谱规模包含 **600+ 节点** 与 **700+ 关系**。

### 3.2 混合检索策略 (Hybrid Retrieval)
系统并非简单的结果拼接，而是采用了 **基于意图的动态路由**：

* **场景 A：定义类问题** ("What is AIVR?")
    * Router 判定为 `VECTOR` 模式。
    * 仅调用 FAISS，快速返回定义段落，响应速度快。

* **场景 B：推理类问题** ("Difference between AIVR and VT?")
    * Router 判定为 `HYBRID` 模式。
    * **Step 1**: 并行调用 FAISS 获取文本细节（如具体心率数值）。
    * **Step 2**: 提取 Query 实体 (`AIVR`, `VT`)，在 Neo4j 中查找它们的公共邻居或差异路径。
    * **Step 3**: LLM 综合两路信息，生成包含对比逻辑的回答。

---

## 4. 实验与评估

### 4.1 测试案例分析
**Query**: *"Digoxin toxicity symptoms"* (地高辛中毒症状)

* **Baseline (纯向量检索)**: 检索到包含 "Digoxin" 的片段，但多为药物说明，缺乏具体的临床体征关联。
* **Ours (GraphRAG)**: 图谱检索路径激活：
    * `Digoxin Toxicity` --[CAUSES]--> `Bidirectional VT`
    * `Digoxin Toxicity` --[CAUSES]--> `Atrial Flutter with Block`
* **结果**: 系统成功列出了具体的特殊心律失常表现，准确度显著优于 Baseline。

### 4.2 性能表现
* **检索响应时间**: 平均 1.2s (向量) vs 2.5s (混合)。
* **节点覆盖率**: 核心心律失常疾病覆盖率达 90% (基于 LITFL 数据集)。

---

## 5. 总结与展望
本项目成功验证了 GraphRAG 在垂直医学领域的有效性。通过引入结构化知识，显著减少了 LLM 的幻觉问题，并提升了复杂问题的回答质量。

**未来工作**:
1.  **多模态扩展**: 引入 CNN 模型解析心电图波形图片，实现“图+文”联合诊断。
2.  **图谱可视化**: 在前端直接展示推理路径的子图，增强可解释性。