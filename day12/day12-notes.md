 Day 12 笔记

## 短期记忆
- messages 列表
- LangGraph 用 MemorySaver
- thread_id 区分会话

## 长期记忆
- 向量库存重要信息
- 检索时找相关信息
- 拼接进 prompt

## 自动提取
- 让 AI 判断哪些信息值得记住
- 提取后存入向量库

## 完整记忆 Agent
- chat 节点：检索记忆 + 回答
- save 节点：提取 + 存储
- 两个节点循环

## 踩的坑
1. 提取记忆的 prompt 要清晰
2. 避免存储重复信息
3. 向量库检索 k 值要合理