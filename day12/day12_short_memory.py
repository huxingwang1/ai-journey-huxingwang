import sys
import io
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 1. 状态 ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ========== 2. 节点 ==========
def chat_node(state: State):
    messages = [SystemMessage(content="你是一个AI助手，记住用户说的话。")] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# ========== 3. 构建图 ==========
workflow = StateGraph(State)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

# ========== 4. 加短期记忆 ==========
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ========== 5. 测试 ==========
config = {"configurable": {"thread_id": "user1"}}

print("第1轮：")
result = app.invoke({"messages": [HumanMessage(content="我叫张三.")]}, config=config)
print(result["messages"][-1].content)

print("\n第2轮：")
result = app.invoke({"messages": [HumanMessage(content="我叫什么名字？")]}, config=config)
print(result["messages"][-1].content)