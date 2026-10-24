from fastapi import FastAPI
from pydantic import BaseModel
from agent.graph import agent_app

app = FastAPI(title="Multi-Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        result = agent_app.invoke({"question": request.message})
        return {"reply": result["answer"]}
    except Exception as e:
        # 重点：强制打印红色错误
        import traceback
        traceback.print_exc()
        return {"reply": f"服务器内部报错：{str(e)}"}
    
    