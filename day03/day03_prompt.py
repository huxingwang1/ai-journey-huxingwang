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

# 实验1：没有 system prompt
messages1 = [
    {"role": "user", "content": "介绍一下Python"}
]
print("=== 无 system ===")
print(chat(messages1))

# 实验2：有 system prompt
messages2 = [
    {"role": "system", "content": "你是一个面向小学生的编程老师，用最简单的话解释，每句话不超过20个字。"},
    {"role": "user", "content": "介绍一下Python"}
]
print("\n=== 有 system（小学生老师）===")
print(chat(messages2))

# 实验3：换一个 system prompt
messages3 = [
    {"role": "system", "content": "你是一个资深Python架构师，用专业术语回答，面向有10年经验的开发者。"},
    {"role": "user", "content": "介绍一下Python"}
]
print("\n=== 有 system（架构师）===")
print(chat(messages3))