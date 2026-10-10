import sys
import io
import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

class State(TypedDict):
    question: str
    draft: str
    approved: bool
    final: str

def draft_node(state: State):
    response = llm.invoke(f"起草一个回复：{state['question']}")
    print(f"[AI 起草] {response.content}")
    return {"draft": response.content}

def review_node(state: State):
    print("\n[人工审核] 请审核以下草稿：")
    print(state["draft"])
    return {"approved": True}

def finalize_node(state: State):
    if state["approved"]:
        return {"final": f"已批准：{state['draft']}"}
    else:
        return {"final": "审核未通过，已退回"}

workflow = StateGraph(State)
workflow.add_node("draft", draft_node)
workflow.add_node("review", review_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("draft")
workflow.add_edge("draft", "review")
workflow.add_edge("review", "finalize")
workflow.add_edge("finalize", END)

# ========== 替换部分 ==========
memory = MemorySaver()

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["review"]
)

config = {"configurable": {"thread_id": "1"}}
result = app.invoke({"question": "客户投诉物流太慢"}, config=config)
print("等待人工审核...")

result = app.invoke(None, config=config)
print(f"最终：{result['final']}")