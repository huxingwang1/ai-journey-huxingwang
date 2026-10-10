import sys
import io
import os
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
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

# ========== 1. 定义状态 ==========
class State(TypedDict):
    question: str
    category: str
    answer: str

# ========== 2. Supervisor 节点 ==========
def supervisor_node(state: State):
    """主管：判断问题属于哪类"""
    system = """你是一个客服主管。根据用户的问题，决定交给哪个子 Agent：
- knowledge：公司政策、产品问题
- order：订单查询
- weather：天气查询
- calculator：数学计算

只回答一个词：knowledge/order/weather/calculator"""
    
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["question"])
    ])
    
    category = response.content.strip().lower()
    # 规范化
    for valid in ["knowledge", "order", "weather", "calculator"]:
        if valid in category:
            category = valid
            break
    else:
        category = "knowledge"
    
    print(f"[Supervisor] 路由到：{category}")
    return {"category": category}

# ========== 3. 子 Agent 节点 ==========
def knowledge_node(state: State):
    print("[知识库] 处理中...")
    response = llm.invoke(f"用知识库回答：{state['question']}")
    return {"answer": response.content}

def order_node(state: State):
    print("[订单] 处理中...")
    response = llm.invoke(f"模拟查询订单：{state['question']}")
    return {"answer": response.content}

def weather_node(state: State):
    print("[天气] 处理中...")
    response = llm.invoke(f"模拟查询天气：{state['question']}")
    return {"answer": response.content}

def calculator_node(state: State):
    print("[计算] 处理中...")
    response = llm.invoke(f"模拟计算：{state['question']}")
    return {"answer": response.content}

# ========== 4. 路由函数 ==========
def route(state: State) -> Literal["knowledge", "order", "weather", "calculator"]:
    return state["category"]

# ========== 5. 构建图 ==========
workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("knowledge", knowledge_node)
workflow.add_node("order", order_node)
workflow.add_node("weather", weather_node)
workflow.add_node("calculator", calculator_node)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route,
    {
        "knowledge": "knowledge",
        "order": "order",
        "weather": "weather",
        "calculator": "calculator"
    }
)

for node in ["knowledge", "order", "weather", "calculator"]:
    workflow.add_edge(node, END)

app = workflow.compile()

# ========== 6. 测试 ==========
if __name__ == "__main__":
    questions = [
        "公司年假政策是什么？",
        "查一下订单 ORD001",
        "北京天气怎么样？",
        "计算 123 * 456"
    ]
    for q in questions:
        print(f"\n{'='*40}")
        print(f"问题：{q}")
        result = app.invoke({"question": q})
        print(f"回答：{result['answer']}")