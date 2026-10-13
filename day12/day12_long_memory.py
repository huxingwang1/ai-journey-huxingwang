import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

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

# 长期记忆库
memory_store = Chroma(
    collection_name="long_term_memory",
    embedding_function=embeddings
)

def save_memory(text):
    """保存记忆"""
    memory_store.add_texts([text])
    print(f"[存储记忆] {text}")

def retrieve_memory(query, k=3):
    """检索记忆"""
    results = memory_store.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in results])

def chat(question):
    # 1. 检索记忆
    memories = retrieve_memory(question)
    
    # 2. 拼接
    messages = [
        SystemMessage(content=f"你是一个AI助手。以下是你记得的关于用户的信息：\n{memories}"),
        HumanMessage(content=question)
    ]
    
    # 3. 生成回答
    response = llm.invoke(messages)
    return response.content

# 测试
print("=== 第一次对话：告诉AI信息 ===")
save_memory("用户的名字是张三")
save_memory("用户喜欢喝咖啡")
save_memory("用户住在杭州")

print("\n=== 第二次对话：问AI ===")
print(chat("我叫什么名字？"))
print(chat("我喜欢喝什么？"))
print(chat("我住在哪里？"))