import sys
import io
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

app = FastAPI(title="Agent API")

# ========== 1. 初始化 LLM ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 2. 构建 RAG 知识库 ==========
print("正在构建知识库..")

embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

pdf_path = r"C:\Users\Administrator\Desktop\ai-journey\Artificial Intelligence.pdf"
full_text = parse_pdf(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_text(full_text)
print(f"切片完成：{len(chunks)} 段")

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="day10_api_rag"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("知识库构建完成")

# ========== 3. 定义工具 ==========
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

@tool
def search_knowledge(question: str) -> str:
    """从知识库中搜索信息。用于回答关于人工智能、教学机智的问题。"""
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs)

tools = [get_weather, calculator, search_knowledge]
llm_with_tools = llm.bind_tools(tools)

# ========== 4. 定义状态 ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ========== 5. 定义节点 ==========
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

# ========== 6. 构建图 ==========
workflow = StateGraph(State)
workflow.add_node("think", think_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("think")
workflow.add_conditional_edges("think", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "think")

agent_app = workflow.compile()

# ========== 7. API 接口 ==========
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"message": "Agent API 已启动"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent_app.invoke({
        "messages": [HumanMessage(content=request.message)]
    })
    reply = result["messages"][-1].content
    return ChatResponse(reply=reply)