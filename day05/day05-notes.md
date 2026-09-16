# Day 5 笔记

## RAG 完整链路
1. PDF 解析（pypdf）
2. 切片（chunk_size=200, overlap=50）
3. 向量化（智谱 embedding-3）
4. 存入 Chroma
5. 问题向量化
6. 检索相似片段
7. 拼接问题+资料
8. DeepSeek 生成答案

## 关键参数
- chunk_size=200：每段200字
- overlap=50：重叠50字，避免切断语义
- n_results=3：检索3条最相关
- batch_size=20：分批存入

## 关键认知
- 好的 Embedding 模型决定检索质量
- system prompt 要求 AI 不编造
- 资料里没有就回答"没有相关信息"

## 实际效果
- PDF：173段
- 3个问题，回答全部精准
- AI 完全基于 PDF 内容回答

## 踩的坑
1. help.pdf 没加引号
2. pypdf 解析 PDF 可能丢格式
3. 切片太短丢上下文，太长超token.