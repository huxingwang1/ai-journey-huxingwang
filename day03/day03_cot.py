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

# 实验1：直接问
messages1 = [
    {"role": "user", "content": "小明比小红大3岁，小红比小刚大2岁，小刚今年10岁。小明今年几岁？"}
]
print("=== 直接问 ===")
print(chat(messages1))

# 实验2：加 CoT
messages2 = [
    {"role": "user", "content": "小明比小红大3岁，小红比小刚大2岁，小刚今年10岁。小明今年几岁？请一步步推理。"}
]
print("\n=== CoT ===")
print(chat(messages2))