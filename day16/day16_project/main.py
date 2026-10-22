from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent.graph import agent_app

app = FastAPI(title="Modular Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.session_id}}
    result = agent_app.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config
    )
    return {"reply": result["messages"][-1].content}