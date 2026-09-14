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

# 实验1：没有例子
messages1 = [
    {"role": "system", "content": "把用户的话翻译成英文。"},
    {"role": "user", "content": "今天天气真好"}
]
print("=== 无例子 ===")
print(chat(messages1))

# 实验2：有例子（Few-shot）
messages2 = [
    {"role": "system", "content": "把用户的话翻译成英文。"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "谢谢"},
    {"role": "assistant", "content": "Thank you"},
    {"role": "user", "content": "今天天气真好"}
]
print("\n=== 有例子（Few-shot）===")
print(chat(messages2))