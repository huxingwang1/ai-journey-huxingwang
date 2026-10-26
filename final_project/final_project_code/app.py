import streamlit as st
import requests
import os

# 1. 动态获取 API 地址
# 如果在 Docker Compose 环境中，会读取到环境变量 API_URL（如 http://ai-api:8000/chat）
# 如果在本地直接运行，默认使用 http://127.0.0.1:8000/chat
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

st.set_page_config(page_title="AI 智能助理", page_icon="🤖")

st.title("🤖 企业级多智能体客服")
st.markdown("基于 LangGraph + Supervisor 架构 + Redis 持久化记忆")

# 初始化历史记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        # 如果历史消息里包含转人工标记，也展示出来
        if msg.get("need_human"):
            st.error("🚨 历史记录：曾触发人工客服介入。")

# 接收用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    # 2. 请求后端 API
    with st.chat_message("assistant"):
        with st.spinner("Agent 思考中..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"message": prompt, "session_id": "streamlit_user"},
                    timeout=30 # 增加超时限制，防止后端卡死前端一直转圈
                )
                
                # 检查后端是否正常返回
                if response.status_code == 200:
                    data = response.json()
                    reply = data.get("reply", "出错了")
                    need_human = data.get("need_human", False)
                else:
                    reply = f"后端报错！状态码：{response.status_code}，内容：{response.text}"
                    need_human = False
                    
            except requests.exceptions.ConnectionError:
                reply = "无法连接后端服务，请检查后端是否启动或端口是否正确。"
                need_human = False
            except Exception as e:
                reply = f"前端发生未知错误：{str(e)}"
                need_human = False
                
            # 3. 打印回复
            st.write(reply)
            
            # 4. 如果触发了转人工，弹出红色警告框
            if need_human:
                st.error("🚨 系统已检测到您的强烈诉求，正在为您接入人工客服，请稍候...")
                
    # 5. 保存助手回复到历史记录
    st.session_state.messages.append({
        "role": "assistant", 
        "content": reply,
        "need_human": need_human
    })