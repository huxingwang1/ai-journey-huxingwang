import sys
import io
import os
import requests
from dotenv import load_dotenv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

def rerank(query, documents, top_k=3):
    """用智谱 Rerank 精排。"""
    response = requests.post(
        "https://open.bigmodel.cn/api/paas/v4/rerank",
        headers={
            "Authorization": f"Bearer {os.getenv('ZHIPU_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "rerank",
            "query": query,
            "documents": documents
        }
    )
    results = response.json()["results"]
    top_docs = [documents[item["index"]] for item in results[:top_k]]
    return top_docs

# 测试
from day13_hybrid import vectorstore

query = "教学机智在感知层面有什么新特征？"
docs = [doc.page_content for doc in vectorstore.similarity_search(query, k=10)]

print("=== Rerank 前 ===")
for doc in docs[:3]:
    print(f"- {doc[:80]}...")

print("\n=== Rerank 后 ===")
top_docs = rerank(query, docs, top_k=3)
for doc in top_docs:
    print(f"- {doc[:80]}...")