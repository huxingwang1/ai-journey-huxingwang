import sys
import io
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

app = FastAPI(title="Memory Agent API")

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

memory_store = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def memory_chat_node(state: State):
    last_message = state["messages"][-1]
    question = last_message.content
    
    memories = memory_store.similarity_search(question, k=3)
    memory_text = "\n".join([doc.page_content for doc in memories]) if memories else "暂无相关记忆"
    
    messages = [
        SystemMessage(content=f"你是一个有记忆的AI助手。你记得：\n{memory_text}"),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

def save_memory_node(state: State):
    messages = state["messages"]
    if len(messages) >= 2:
        question = messages[-2].content
        answer = messages[-1].content
        
        prompt = f"""从下面的对话中提取值得长期记住的信息,（如姓名、喜好、地址等）。
如果没有，回答"无"。
用户：{question}
AI：{answer}
只输出要记住的信息。"""
        
        response = llm.invoke(prompt)
        memory = response.content.strip()
        
        if memory and memory != "无":
            memory_store.add_texts([memory])
            print(f"[存储记忆] {memory}")
    
    return {}

workflow = StateGraph(State)
workflow.add_node("chat", memory_chat_node)
workflow.add_node("save", save_memory_node)

workflow.set_entry_point("chat")
workflow.add_edge("chat", "save")
workflow.add_edge("save", END)

agent_app = workflow.compile()

# ========== API 接口 ==========
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent_app.invoke({"messages": [HumanMessage(content=request.message)]})
    reply = result["messages"][-1].content
    return ChatResponse(reply=reply)