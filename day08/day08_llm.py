import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 单次调用
response = llm.invoke("请用一句话介绍你自己")
print("单次调用：")
print(response.content)

# 多轮对话
messages = [
    SystemMessage(content="你是一个专业的AI助手。"),
    HumanMessage(content="我叫张三"),
    AIMessage(content="你好，张三！"),
    HumanMessage(content="我叫什么名字？")
]

response = llm.invoke(messages)
print("\n多轮对话：")
print(response.content)