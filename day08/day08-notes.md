Day 8 笔记

## LangChain 核心概念
- LLM：`ChatOpenAI`，用 `llm.invoke()`
- PromptTemplate：`ChatPromptTemplate.from_messages()`
- Chain：`prompt | llm | StrOutputParser()`
- LCEL：用 `|` 管道符组合

## LangChain RAG
- `OpenAIEmbeddings` 替代手写 Embedding
- `RecursiveCharacterTextSplitter` 替代手写切片
- `Chroma.from_texts()` 替代手动 add
- `vectorstore.as_retriever()` 替代手动 query
- `RunnablePassthrough()` 传问题
- `chain.invoke()` 执行

## 对比
- 手写：理解原理
- LangChain：提高效率
- 两者都要会

## 踩的坑
1. `langchain.text_splitter` 改成 `langchain_text_splitters`
2. 每个文件要单独定义 `llm`
3. LangChain 版本更新快，注意导入路径