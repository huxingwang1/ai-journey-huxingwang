import sys
import io
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from day14_stream import vectorstore



load_dotenv()

# ========== 1. 初始化 RAG 链 ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 确保这里使用的 collection_name 是你之前建好的库
vectorstore = Chroma(collection_name="hybrid_test", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识库助手，只能根据提供的资料回答问题，不要编造。"),
    ("user", "参考资料：\n{context}\n\n问题：{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ========== 2. 启动 FastAPI ==========
app = FastAPI(title="RAG Stream API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG Stream API 已启动"}

@app.post("/query/stream")
def query_stream(request: QueryRequest):
    """
    使用流式输出。
    FastAPI 的 StreamingResponse 需要一个生成器 (yield)
    """
    def generate():
        # 核心：使用 .stream() 而不是 .invoke()
        for chunk in rag_chain.stream(request.question):
            # SSE (Server-Sent Events) 的标准格式：数据前面加 "data: "，结尾加两个换行 "\n\n"
            yield f"data: {chunk}\n\n"
            
    # 注意 media_type 必须是 text/event-stream
    return StreamingResponse(generate(), media_type="text/event-stream")