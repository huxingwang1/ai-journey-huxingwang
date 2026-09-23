import sys
import io
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

app = FastAPI(title="LangChain RAG API")

# 1. 初始化
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

# 2. 加载 PDF
def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

print("正在加载 PDF...")
full_text = parse_pdf("Artificial Intelligence.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_text(full_text)
print(f"切片完成：{len(chunks)} 段")

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="langchain_rag"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
 
# 3. 提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识库助手。只能根据提供的资料回答问题，不要编造。"),
    ("user", "参考资料：\n{context}\n\n问题：{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. 构建 RAG 链
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. API 接口
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"message": "LangChain RAG API 已启动"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    answer = rag_chain.invoke(request.question)
    return QueryResponse(answer=answer)