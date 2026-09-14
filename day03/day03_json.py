import sys
import io
import os
import json
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

# 实验1：不要求 JSON
messages1 = [
    {"role": "user", "content": "提取这句话里的人名和年龄：张三今年25岁，李四今年30岁。"}
]
print("=== 不要求 JSON ===")
print(chat(messages1))

# 实验2：要求 JSON
messages2 = [
    {"role": "system", "content": "你是一个信息提取助手。只输出JSON，不要任何其他文字。格式：{\"people\": [{\"name\": \"\", \"age\": 0}]}"},
    {"role": "user", "content": "提取这句话里的人名和年龄：张三今年25岁，李四今年30岁。"}
]
print("\n=== 要求 JSON ===")
result = chat(messages2)
print(result)

# 解析 JSON
try:
    data = json.loads(result)
    for person in data["people"]:
        print(f"{person['name']}：{person['age']}岁")
except json.JSONDecodeError:
    print("解析失败，AI 没有输出合法 JSON")