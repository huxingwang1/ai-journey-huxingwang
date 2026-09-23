import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 1. 定义状态
class State(TypedDict):
    question: str
    answer: str

# 2. 定义节点
def answer_node(state: State):
    response = llm.invoke(state["question"])
    return {"answer": response.content}

# 3. 构建图
workflow = StateGraph(State)
workflow.add_node("answer", answer_node)
workflow.set_entry_point("answer")
workflow.add_edge("answer", END)

# 4. 编译
app = workflow.compile()

# 5. 运行
result = app.invoke({"question": "什么是人工智能？"})
print("回答：", result["answer"])