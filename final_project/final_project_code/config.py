import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import redis

load_dotenv()

# 大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    timeout=15,        # 15秒超时
    max_retries=2      # 失败重试2次
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


# 1. 指定一个固定的本地文件夹（比如 chroma_db）
CHROMA_PATH = "./chroma_db"

# 2. 连接向量库时，加上 persist_directory 参数
knowledge_store = Chroma(
    collection_name="hybrid_test", 
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH  # <--- 核心：告诉它存到硬盘上
)

# 3. 自动建库逻辑改为：只在文件夹为空时才建
if not os.path.exists(CHROMA_PATH) or knowledge_store._collection.count() == 0:
    print("⚠️ 检测到本地向量库为空，正在调用 Embedding API 构建（仅限首次）...")
    # ... 这里放你读取 PDF 和存入数据的代码 ...
    # knowledge_store.add_texts(chunks)
    print("✅ 建库完成！下次启动将秒开。")
else:
    print("🚀 检测到本地已有向量库，直接加载，无需重新 Embedding！")

pdf_path = os.path.join("Artificial Intelligence.pdf")

reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() or ""
            
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    knowledge_store.add_texts(chunks)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0, decode_responses=True)