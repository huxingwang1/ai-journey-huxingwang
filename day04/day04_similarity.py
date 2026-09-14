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

# 测试
texts = [
    "我喜欢猫",
    "我爱猫咪",
    "今天天气真好",
    "猫是一种可爱的动物"
]

# 获取所有向量
vectors = [get_embedding(t) for t in texts]

# 两两比较
print("相似度矩阵：")
for i, t1 in enumerate(texts):
    for j, t2 in enumerate(texts):
        if i < j:
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"{t1} vs {t2}：{sim:.4f}")