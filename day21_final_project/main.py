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
    print(f"\n🔥🔥🔥 收到前端请求：{request.message} 🔥🔥🔥\n")
    
    # 1. 第一道防线：输入护栏
    if not filter_input(request.message):
        print(f"\n🚨 [拦截] 触发敏感词：{request.message}\n")
        return {"reply": "您的输入包含违规或敏感指令，已被系统拦截。"}
        
    # 2. 第二道防线：查缓存（放在护栏之后）
    if request.message in qa_cache:
        print(f"\n🚀 [命中缓存] 直接返回，省下大模型的钱！\n")
        return {"reply": qa_cache[request.message]}

    # 3. 防死循环配置
    config = {
        "configurable": {"thread_id": request.session_id},
        "recursion_limit": 10
    }

    try:
        # 4. 运行 Agent
        result = agent_app.invoke({"question": request.message}, config=config)
        reply = result["answer"]
        
        # 5. 输出护栏
        safe_reply = filter_output(reply)
        
        # 6. 存入缓存
        qa_cache[request.message] = safe_reply
        
        return {"reply": safe_reply}

    # 7. 异常兜底
    except GraphRecursionError:
        print(f"\n🚨 [报错] Agent 触发死循环\n")
        return {"reply": "抱歉，Agent 思考次数过多，已强制停止。"}
    except Exception as e:
        print(f"\n🚨 [报错] 系统异常：{e}\n")
        return {"reply": f"服务器开小差了：{str(e)}"}