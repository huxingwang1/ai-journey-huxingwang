import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件里的环境变量
load_dotenv()

# 从环境变量里读密钥
api_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "你好，请用一句话介绍你自己"}
    ]
)

print(response.choices[0].message.content)

