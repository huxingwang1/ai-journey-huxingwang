import sys
import io
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# ========== 1. 初始化 LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 2. 定义工具 ==========
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    weather_data = {
        "北京": "晴，25°C",
        "上海": "多云，28°C",
        "杭州": "小雨，22°C"
    }
    return weather_data.get(city, f"没有{city}的天气数据")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"

# ========== 3. 导入 RAG 工具 ==========
sys.path.append("Artificial Intelligence.pdf")
from day09_rag_api import rag_chain

@tool
def search_knowledge(question: str) -> str:
    """从知识库中搜索信息。用于回答关于人工智能、教学机智的问题。"""
    return rag_chain.invoke(question)

# ========== 4. 绑定工具 ==========
tools = [get_weather, calculator, search_knowledge]
llm_with_tools = llm.bind_tools(tools)

# ========== 5. 定义状态 ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ========== 6. 定义节点 ==========
def think_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: State):
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"[调用工具] {tool_name}({tool_args})")
        
        tool_map = {
            "get_weather": get_weather,
            "calculator": calculator,
            "search_knowledge": search_knowledge
        }
        result = tool_map[tool_name].invoke(tool_args)
        results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# ========== 7. 构建图 ==========
workflow = StateGraph(State)
workflow.add_node("think", think_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("think")
workflow.add_conditional_edges("think", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "think")

app = workflow.compile()

# ========== 8. 测试 ==========
if __name__ == "__main__":
    questions = [
        "北京天气怎么样？",
        "计算 123 * 456",
        "教学机智在感知层面有什么新特征？"
    ]
    for q in questions:
        print(f"\n{'='*40}")
        print(f"问题：{q}")
        result = app.invoke({"messages": [HumanMessage(content=q)]})
        print(f"最终回答：{result['messages'][-1].content}")