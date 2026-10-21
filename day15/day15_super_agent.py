import sys
import io
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

pdf_path = r"Artificial Intelligence.pdf"
full_text = parse_pdf(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_text(full_text)
print(f"切片完成：{len(chunks)} 段")

# ========== 1. 基础初始化 ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 长期记忆库和知识库
memory_store = Chroma(collection_name="agent_long_memory", embedding_function=embeddings)
knowledge_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="hybrid_test"
)
knowledge_retriever = knowledge_store.as_retriever(search_kwargs={"k": 3})

# ========== 2. 定义工具 ==========
@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city}今天晴，25°C。"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return f"结果是：{eval(expression)}"
    except Exception as e:
        return f"错误：{e}"

@tool
def search_knowledge(question: str) -> str:
    """从知识库搜索信息。用于回答关于人工智能、教学机智的问题。"""
    docs = knowledge_retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])

tools = [get_weather, calculator, search_knowledge]
llm_with_tools = llm.bind_tools(tools)
# ========== 3. 定义状态 ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ========== 4. 定义节点 ==========
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
    """ReAct 思考节点"""
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
    """判断是否继续调用工具"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "save"

# ========== 5. 构建图 ==========
workflow = StateGraph(State)
workflow.add_node("memory", memory_retrieval_node)
workflow.add_node("think", think_node)
workflow.add_node("tools", tool_node)
workflow.add_node("save", save_memory_node)

workflow.set_entry_point("memory")
workflow.add_edge("memory", "think")
workflow.add_conditional_edges("think", should_continue, {"tools": "tools", "save": "save"})
workflow.add_edge("tools", "think")
workflow.add_edge("save", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ========== 6. 测试 ==========
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user1"}}
    
    q1 = "我叫李四，在杭州做程序员。"
    print(f"用户：{q1}")
    res = app.invoke({"messages": [HumanMessage(content=q1)]}, config=config)
    print(f"AI：{res['messages'][-1].content}\n")
    
    q2 = "帮我查一下杭州的天气。"
    print(f"用户：{q2}")
    res = app.invoke({"messages": [HumanMessage(content=q2)]}, config=config)
    print(f"AI：{res['messages'][-1].content}\n")
    
    q3 = "你还记得我是做什么工作的吗？"
    print(f"用户：{q3}")
    res = app.invoke({"messages": [HumanMessage(content=q3)]}, config=config)
    print(f"AI：{res['messages'][-1].content}")