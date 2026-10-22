from config import llm, memory_store
from agent.state import State
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from tools.weather import get_weather
from tools.calculator import calculator
from tools.rag import search_knowledge
from langgraph.graph import END

# 绑定工具
tools = [get_weather, calculator, search_knowledge]
llm_with_tools = llm.bind_tools(tools)

def memory_retrieval_node(state: State):
    # ...(复制你昨天的记忆检索逻辑，记得用兼容字典的写法)...
    pass

def think_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: State):
    # ...(复制你昨天的工具执行逻辑)...
    pass

def save_memory_node(state: State):
    # ...(复制你昨天的存储逻辑)...
    pass


def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "save"