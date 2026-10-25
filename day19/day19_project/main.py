import sys
import io
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.errors import GraphRecursionError
from agent.graph import agent_app
from guardrails import filter_input, filter_output

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

load_dotenv()

app = FastAPI(title="Secure Multi-Agent API")

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
def chat(request: ChatRequest):
    print("\n【1】收到请求，准备过护栏...", flush=True) # 加上 flush=True
    
    if not filter_input(request.message):
        return {"reply": "拦截"}
        
    print("【2】过了护栏，准备进入图...", flush=True)
    
    try:
        # 注意：这里的线程可能卡住
        print("【3】即将执行 invoke...", flush=True)
        result = agent_app.invoke({"question": request.message})
        print("【4】图执行完毕，拿到结果了！", flush=True)
        
        reply = result["answer"]
        return {"reply": reply}
    except Exception as e:
        print(f"【报错】{e}", flush=True)
        return {"reply": "出错了"}

   