import sys
import io
import os
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

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

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

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识库助手。只能根据提供的资料回答问题，不要编造。"),
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

questions = [
    "教学机智在感知层面有什么新特征？",
    "教师如何保持教学机智？"
]

for q in questions:
    print(f"\n问题：{q}")
    print(f"回答：{rag_chain.invoke(q)}")