from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import agent_app

app = FastAPI(title="Multi-Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
def chat(request: ChatRequest):
    # 注意：Day 17 的状态传入的是 question
    result = agent_app.invoke({"question": request.message})
    
    # 注意：Day 17 的专家节点返回的是 answer
    return {"reply": result["answer"]}