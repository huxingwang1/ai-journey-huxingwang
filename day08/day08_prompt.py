import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# ========== 加上这段 ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
# ==============================

# 定义模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，用{style}的风格回答。"),
    ("user", "{question}")
])

# 多种组合
configs = [
    {"role": "小学生编程老师", "style": "简单易懂", "question": "什么是Python？"},
    {"role": "资深架构师", "style": "专业严谨", "question": "什么是Python？"},
]

for config in configs:
    messages = prompt.format_messages(**config)
    response = llm.invoke(messages)
    print(f"=== {config['role']} ===")
    print(response.content)
    print()