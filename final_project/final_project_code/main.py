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

qa_cache = {}
@app.post("/chat")
def chat(request: ChatRequest):
    # 1. 输入护栏
    if not filter_input(request.message):
        return {"reply": "您的输入包含违规或敏感指令，已被系统拦截。", "need_human": False}

    # 2. 查缓存（注意：如果上次是转人工，尽量不走缓存，这里为了简单，统一走缓存）
    if request.message in qa_cache:
        print(f"\n🚀 [命中缓存] 直接返回，省下大模型的钱！\n")
        return qa_cache[request.message] # 缓存里存的是整个字典，包含 reply 和 need_human

    config = {
        "configurable": {"thread_id": request.session_id},
        "recursion_limit": 10
    }

    try:
        # 3. 运行 Agent
        result = agent_app.invoke({"question": request.message}, config=config)
        reply = result.get("answer", "系统无法回答")
        need_human = result.get("need_human", False) 
        
        safe_reply = filter_output(reply)
        
        # 组装最终的响应体
        response_data = {"reply": safe_reply, "need_human": need_human}
        
        # 4. 存入缓存
        qa_cache[request.message] = response_data
        
        return response_data

    except GraphRecursionError:
        return {"reply": "抱歉，Agent 思考次数过多，已强制停止。", "need_human": False}
    except Exception as e:
        print(f"\n🚨 [报错] 系统异常：{e}\n")
        return {"reply": f"服务器开小差了：{str(e)}", "need_human": False}