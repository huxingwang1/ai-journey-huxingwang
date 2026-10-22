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
    """在思考和回答前，先检索长期记忆并注入"""
    last_message = state["messages"][-1] 
    if isinstance(last_message, HumanMessage):
        question = last_message.content
        # 检索长期记忆
        memories = memory_store.similarity_search(question, k=2)
        if memories:
            memory_text = "\n".join([doc.page_content for doc in memories])
            sys_msg = SystemMessage(content=f"你记得关于用户的信息：\n{memory_text}")
            return {"messages": [sys_msg]}
    return {}


def think_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: State):
    """执行工具"""
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        # 兼容字典和对象的写法
        tool_name = tool_call["name"] if isinstance(tool_call, dict) else tool_call.name
        tool_args = tool_call["args"] if isinstance(tool_call, dict) else tool_call.args
        tool_id = tool_call["id"] if isinstance(tool_call, dict) else tool_call.id
        
        print(f"[调用工具] {tool_name}({tool_args})")
        
        tool_map = {
            "get_weather": get_weather,
            "calculator": calculator,
            "search_knowledge": search_knowledge
        }
        result = tool_map[tool_name].invoke(tool_args)
        results.append(ToolMessage(content=result, tool_call_id=tool_id))
    return {"messages": results}



def save_memory_node(state: State):
    """对话结束后，自动提取记忆存入长期库"""
    messages = state["messages"]
    if len(messages) >= 2:
        human_msg = [m for m in messages if isinstance(m, HumanMessage)][-1].content
        ai_msg = messages[-1].content if messages[-1].content else ""
        
        if ai_msg:
            prompt = f"从对话提取用户个人信息，没有则回'无'。\n用户：{human_msg}\nAI：{ai_msg}"
            res = llm.invoke(prompt).content.strip()
            if res and res != "无":
                memory_store.add_texts([res])
                print(f"[存储长期记忆] {res}")
    return {}


def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "save"