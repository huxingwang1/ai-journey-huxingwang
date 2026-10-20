Day 14 笔记

## 流式输出 (Streaming)
- `chain.stream()` 返回的是生成器，需要 `for` 循环 + `flush=True`。
- 极大提升用户体验。

## 上下文压缩 (Compression)
- 为了节省 token 和防幻觉，用 LLM 提取文档中与问题相关的句子。
- 用 `.batch()` 并发处理，避免串行太慢。

## 语义缓存 (Semantic Cache)
- 用向量库存历史问答。
- 用 `similarity_search_with_score` 查，分数达标直接返回。
- 省钱利器。

## 异步并发 (Async)
- `await chain.ainvoke(...)`。
- `asyncio.gather` 处理并发请求。