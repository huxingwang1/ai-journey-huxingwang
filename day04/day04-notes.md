Day 4 笔记

## Embedding
- 把文字变成向量（一串数字）
- 语义相似的文本，向量距离近
- 智谱 embedding-3 输出 2048 维

## 余弦相似度
- 计算两个向量的相似程度
- 范围：-1 到 1，越接近 1 越相似
- 公式：点积 / (模长 × 模长)

## 语义 vs 关键词
- "如何申请年假" vs "休假需要什么流程"：0.78
- 证明 Embedding 捕捉的是语义，不是关键词

## Chroma
- 向量数据库，存储和检索向量
- 默认 Embedding 模型对中文支持差
- 可以自定义 Embedding 函数（用智谱）

## RAG 雏形
- 第一步：检索（Retrieval）→ 找到相关文档
- 第二步：生成（Generation）→ AI 基于文档回答
- 关键：好的 Embedding 模型决定检索质量
- 关键：system prompt 要求 AI 不编造

## 踩的坑
1. ChromaDB 需要 VC++ 运行库
2. 默认英文模型对中文效果差
3. 替换成智谱 Embedding 后效果显著提升
4. n_results 强制返回，可能返回不相关文档