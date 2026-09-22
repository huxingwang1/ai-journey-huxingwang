from fastapi import FastAPI
from pydantic import BaseModel
import sys
import io
import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

app = FastAPI(title="RAG API")

# ========== 初始化 RAG ==========
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

embed_client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def get_embedding(text):
    response = embed_client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding

class ZhipuEmbedding(embedding_functions.EmbeddingFunction):
    def __call__(self, texts):
        return [get_embedding(t) for t in texts]

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

# 启动时构建知识库
print("正在构建知识库...")
full_text = parse_pdf("Artificial Intelligence.pdf")
chunks = split_text(full_text)
print(f"切片完成：{len(chunks)} 段")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="knowledge",
    embedding_function=ZhipuEmbedding()
)

batch_size = 20
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    collection.add(
        documents=batch,
        ids=[f"chunk{j}" for j in range(i, i+len(batch))]
    )
print(f"知识库构建完成：{collection.count()} 段")

# ========== API 接口 ==========
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def root():
    return {"message": "RAG API 已启动"}
@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    # 1. 先检索 10 条候选
    results = collection.query(
        query_texts=[request.question],
        n_results=10
    )
    candidates = results["documents"][0]
    
    # 2. Rerank 精排
    rerank_response = requests.post(
        "https://open.bigmodel.cn/api/paas/v4/rerank",
        headers={
            "Authorization": f"Bearer {os.getenv('ZHIPU_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "rerank",
            "query": request.question,
            "documents": candidates
        }
    ) 
    reranked = rerank_response.json()["results"]
    
    # 3. 取前 3 条最相关的
    top_docs = [candidates[item["index"]] for item in reranked[:3]]
    context = "\n\n".join(top_docs)
    
    # 4. 生成回答
    messages = [
        {"role": "system", "content": "你是一个知识库助手，只能根据提供的资料回答问题，不要编造。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{request.question}"}
    ]
    
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    
    return QueryResponse(
        answer=response.choices[0].message.content,
        sources=top_docs
    )
  