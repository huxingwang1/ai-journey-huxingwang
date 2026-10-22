import sys
import io
import os
import time
from dotenv import load_dotenv
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

# ========== 1. 初始化模型和 Embedding ==========
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


vectorstore = Chroma(collection_name="hybrid_test", embedding_function=embeddings)


pdf_path = r"Artificial Intelligence.pdf"
reader = PdfReader(pdf_path)
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() or ""
        
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50, 
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_text(full_text)
vectorstore.add_texts(chunks)
  

# ========== 3. 构建检索器和链 ==========
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

# ========== 4. 测试流式输出 ==========
print("问题：教学机智在感知层面有什么新特征？")
print("AI：", end="", flush=True)

for chunk in rag_chain.stream("教学机智在感知层面有什么新特征？"):
    print(chunk, end="", flush=True)
print("\n")