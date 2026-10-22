import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


load_dotenv()

# 大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Embedding 和向量库
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


memory_store = Chroma(collection_name="agent_long_memory", embedding_function=embeddings)
knowledge_store = Chroma(collection_name="hybrid_test", embedding_function=embeddings)
knowledge_retriever = knowledge_store.as_retriever(search_kwargs={"k": 3})

pdf_path = os.path.join(os.path.dirname(__file__), "..", "..", "Artificial Intelligence.pdf")
    
if os.path.exists(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
         text += page.extract_text() or ""
            
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    knowledge_store.add_texts(chunks)
   