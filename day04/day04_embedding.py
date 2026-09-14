import sys
import io
import os
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
    """把文本转成向量"""
    response = client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding

# 测试
texts = ["我喜欢猫", "我爱猫咪", "今天天气真好"]

for text in texts:
    vec = get_embedding(text)
    print(f"{text} → 向量长度：{len(vec)}，前5个数字：{vec[:5]}")