import streamlit as st
import requests

st.set_page_config(page_title="AI 智能助理", page_icon="🤖")

st.title("🤖 多智能体 AI 助理")
st.markdown("基于 LangGraph + Supervisor 架构 + Redis 持久化记忆")

# 初始化历史记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 接收用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
    # 2. 请求后端 API
    with st.chat_message("assistant"):
        with st.spinner("Agent 思考中..."):
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={"message": prompt, "session_id": "streamlit_user"}
            )
            # 把前端拦截下来，看看后端到底返回了什么
            if response.status_code == 200:
                reply = response.json().get("reply", "出错了")
            else:
                reply = f"后端报错！状态码：{response.status_code}，内容：{response.text}"
                
            st.write(reply)
    # 3. 保存助手回复
    st.session_state.messages.append({"role": "assistant", "content": reply})