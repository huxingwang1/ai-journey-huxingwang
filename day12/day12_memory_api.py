import sys
import io
import os
from day12_memory_agent import app as agent_app

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# 导入 Agent
from day12_memory_agent import app as agent_app

app = FastAPI(title="Memory Agent API")

# ========== API 接口 ==========
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    return {"message": "Memory Agent API 已启动"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = agent_app.invoke({
        "messages": [HumanMessage(content=request.message)]
    })
    reply = result["messages"][-1].content
    return ChatResponse(reply=reply)