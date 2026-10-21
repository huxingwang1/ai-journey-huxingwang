import sys
import io
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

# 导入 Agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from day15_super_agent import app as agent_app



load_dotenv()

app = FastAPI(title="Super Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str  # 区分用户

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """流式接口"""
    config = {"configurable": {"thread_id": request.session_id}}
    
    def generate():
        # 注意：目前 LangGraph Agent 的流式比较复杂，这里简单模拟直接输出最终结果
        # 真正的 Agent 流式需要用 astream_events，以后会学
        result = agent_app.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config
        )
        reply = result["messages"][-1].content
        # 假装流式（逐字输出）
        for char in reply:
            yield f"data: {char}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")