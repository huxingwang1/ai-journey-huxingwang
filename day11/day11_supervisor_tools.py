import sys
import io
import os
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 1. 定义工具（从 Day 10 复制） ==========
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    weather_data = {
        "北京": "晴，25°C，湿度40%",
        "上海": "多云，28°C，湿度65%",
        "杭州": "小雨，22°C，湿度80%",
        "深圳": "晴，30°C，湿度70%"
    }
    return weather_data.get(city, f"暂时没有{city}的天气数据")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"

@tool
def query_order(order_id: str) -> str:
    """查询订单状态"""
    orders = {
        "ORD001": "已发货，预计明天送达",
        "ORD002": "待付款",
        "ORD003": "已完成"
    }
    return orders.get(order_id, f"未找到订单 {order_id}")

@tool
def search_knowledge(question: str) -> str:
    """从知识库搜索信息"""
    # 这里可以接你 Day 9 的 RAG
    # 暂时用模拟数据
    knowledge = {
        "年假": "入职满1年享受5天年假，满3年10天，满5年15天",
        "报销": "填写报销单，附上发票，提交给直属领导审批",
        "考勤": "上班时间9:00-18:00，迟到超过30分钟扣半天工资"
    }
    for key, value in knowledge.items():
        if key in question:
            return value
    return f"知识库中没有关于「{question}」的信息"

# ========== 2. 定义状态 ==========
class State(TypedDict):
    question: str
    category: str
    answer: str

# ========== 3. Supervisor 节点 ==========
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
    for valid in ["knowledge", "order", "weather", "calculator"]:
        if valid in category:
            category = valid
            break
    else:
        category = "knowledge"
    
    print(f"[Supervisor] 路由到：{category}")
    return {"category": category}

# ========== 4. 子 Agent 节点（调用真实工具） ==========
def knowledge_node(state: State):
    print("[知识库] 处理中...")
    # 简单提取关键词
    question = state["question"]
    result = search_knowledge.invoke({"question": question})
    return {"answer": result}

def order_node(state: State):
    print("[订单] 处理中...")
    # 简单提取订单号
    import re
    match = re.search(r'ORD\d+', state["question"])
    if match:
        order_id = match.group()
        result = query_order.invoke({"order_id": order_id})
    else:
        result = "请提供订单号，如 ORD001"
    return {"answer": result}

def weather_node(state: State):
    print("[天气] 处理中...")
    # 简单提取城市
    cities = ["北京", "上海", "杭州", "深圳"]
    city = "北京"  # 默认
    for c in cities:
        if c in state["question"]:
            city = c
            break
    result = get_weather.invoke({"city": city})
    return {"answer": result}

def calculator_node(state: State):
    print("[计算] 处理中...")
    # 提取表达式
    import re
    match = re.search(r'[\d+\-*/().\s]+', state["question"])
    if match:
        expression = match.group().strip()
        result = calculator.invoke({"expression": expression})
    else:
        result = "请提供数学表达式"
    return {"answer": result}

# ========== 5. 路由函数 ==========
def route(state: State) -> Literal["knowledge", "order", "weather", "calculator"]:
    return state["category"]

# ========== 6. 构建图 ==========
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

# ========== 7. 测试 ==========
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