import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

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