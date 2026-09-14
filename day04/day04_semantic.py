import sys
import io
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def get_embedding(text):
    response = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#测试
texts = [
    "如何申请年假",
    "休假需要什么流程",
    "今天中午吃什么",
    "公司的考勤制度"
]

vectors = [get_embedding(t) for t in texts]

# 比较"如何申请年假"和其他的相似度
query_vec = vectors[0]
print("以'如何申请年假'为基准：")
for i, t in enumerate(texts[1:], 1):
    sim = cosine_similarity(query_vec, vectors[i])
    print(f"  vs {t}：{sim:.4f}")