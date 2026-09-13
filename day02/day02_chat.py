import sys
import io
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# 加载 .env 文件里的环境变量
load_dotenv()

# 从环境变量里读密钥
api_key = os.getenv("DEEPSEEK_API_KEY")


client = OpenAI(
    api_key=api_key,  # 替换成你的
    base_url="https://api.deepseek.com"
)

def chat(messages):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"出错了：{e}"


def save_history(messages, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    

# 初始化对话历史
messages = [
    {"role": "system", "content": "你是一个专业的AI助手，回答简洁明了。"}
]

print("输入 quit 退出")
while True:
    user_input = input("你：")
    if user_input == "quit":
        break
    messages.append({"role": "user", "content": user_input})
    reply = chat(messages)
    messages.append({"role": "assistant", "content": reply})
    print(f"[当前历史消息数：{len(messages)}]")
    print("AI：", reply)
    save_history(messages, "chat_history.json")
