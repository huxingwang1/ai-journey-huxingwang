import sys
import io
import os
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def chat(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    return response.choices[0].message.content

messages1 = [
    {"role": "system", "content": "把用户的话改写成客服话术。"},
    {"role": "user", "content": "这个快递怎么还没到"}
]
print("=== 无例子 ===")
print(chat(messages1))

# 实验2：有例子
messages2 = [
    {"role": "system", "content": "把用户的话改写成客服话术。"},
    {"role": "user", "content": "我要退款"},
    {"role": "assistant", "content": "亲，非常抱歉给您带来不便，我马上为您处理退款事宜，请您稍等哦~"},
    {"role": "user", "content": "这个快递怎么还没到"}
]
print("\n=== 有例子 ===")
print(chat(messages2))