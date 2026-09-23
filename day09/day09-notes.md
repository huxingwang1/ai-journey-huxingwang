 Day 9 笔记

## Memory
- `MessagesPlaceholder`：历史占位
- `RunnableWithMessageHistory`：自动管理历史
- `session_id`：区分用户
- 注意：官方推荐用 LangGraph 管理记忆

## OutputParser
- `JsonOutputParser`：输出 JSON
- `PydanticOutputParser`：用 Pydantic 定义结构
- `parser.get_format_instructions()`：自动生成格式说明
- `prompt.partial(...)`：填充格式说明

## 多链组合
- `RunnableParallel`：并行执行
- `RunnablePassthrough`：原样传递
- `|` 管道符：串联

## LangChain RAG API
- 代码量减少 50%
- `rag_chain.invoke(question)` 一行搞定
- FastAPI 接口极简

## 踩的坑
1. `langchain.text_splitter` 改成 `langchain_text_splitters`
2. 路径必须加引号
3. 每个文件要单独定义 `llm`
4. `RunnableWithMessageHistory` 已被标记为废