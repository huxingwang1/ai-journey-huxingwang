import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

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

# 复用 Day 13 的 vectorstore
from day13_hybrid import vectorstore

def rewrite_query(question):
    """用 LLM 改写问题，生成多个查询"""
    prompt = f"""请把下面的问题改写成 3 个不同角度的查询，每行一个：
{question}"""
    response = llm.invoke(prompt)
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return queries[:3]

def multi_recall(query, k=3):
    """多路召回：原始问题 + 改写问题"""
    all_docs = []
    
    # 1. 原始问题
    results = vectorstore.similarity_search(query, k=k)
    all_docs.extend([doc.page_content for doc in results])
    
    # 2. 改写问题
    rewritten = rewrite_query(query)
    print(f"[改写] {rewritten}")
    for q in rewritten:
        results = vectorstore.similarity_search(q, k=k)
        all_docs.extend([doc.page_content for doc in results])
    
    return list(dict.fromkeys(all_docs))[:k]

# 测试
query = "教学机智在感知层面有什么新特征？"
docs = multi_recall(query, k=3)
for doc in docs:
    print(f"- {doc[:80]}...")