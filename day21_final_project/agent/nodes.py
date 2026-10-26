from config import llm, memory_store
from agent.state import State
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from tools.weather import get_weather
from tools.calculator import calculator
from tools.rag import search_knowledge
from langgraph.graph import END
from langchain_core.prompts import ChatPromptTemplate
from config import llm, knowledge_retriever, redis_client
# 绑定工具
tools = [get_weather, calculator, search_knowledge]
llm_with_tools = llm.bind_tools(tools)

def memory_retrieval_node(state: State):
    """在思考和回答前，先检索长期记忆并注入"""
    last_message = state["messages"][-1] 
    if isinstance(last_message, HumanMessage):
        question = last_message.content
        # 检索长期记忆
        memories = memory_store.similarity_search(question, k=2)
        if memories:
            memory_text = "\n".join([doc.page_content for doc in memories])
            sys_msg = SystemMessage(content=f"你记得关于用户的信息：\n{memory_text}")
            return {"messages": [sys_msg]}
    return {}


# ========== 1. 定义主管提示词 ==========
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个客服主管。根据用户的问题，决定交给哪个专家：
- weather_expert：天气查询
- rag_expert：公司政策、人工智能、教学机智等知识问答
- calc_expert：数学计算
- other_expert：闲聊或其他

请严格只输出专家的英文代号，不要输出任何其他内容。"""),
    ("user", "{question}")
])


def load_memory_node(state: State):
    """从 Redis 读取并保存用户信息"""
    question = state["question"]
    memory_str = ""
    
    # ================= 修复核心：提取名字的防御逻辑 =================
    # 1. 只有句子里包含“我叫”，并且【不包含】“什么/啥/谁”等疑问词时，才去提取名字
    if "我叫" in question and not any(q in question for q in ["什么", "啥", "谁"]):
        import re
        # 2. 限制名字最多匹配 3 个汉字（真实名字很少超过3个字），防止把废话也存进去
        match = re.search(r'我叫([\u4e00-\u9fa5]{1,3})', question)
        if match:
            name = match.group(1)
            # 3. 过滤掉尾部可能多抓的标点
            name = name.strip("，。！？ ")
            if name:
                redis_client.set("user:name", name)
                print(f"[Redis] 已持久化保存用户名：{name}")
    # ================================================================
    
    # 读取逻辑保持不变（每次提问都去查一下）
    name = redis_client.get("user:name")
    if name:
        print(f"[Redis] 命中长期记忆：名字是 {name}")
        memory_str = f"用户的真实名字叫：{name}"
            
    return {"user_memory": memory_str}
# ========== 2. 定义各个专家节点 ==========
def weather_expert(state: State):
    print("  [天气专家] 处理中...")
    # 简单提取城市（这里为了演示，直接假设问题里包含城市）
    city = "北京"
    if "上海" in state["question"]: city = "上海"
    if "杭州" in state["question"]: city = "杭州"
    
    # 调用真实的工具
    result = get_weather.invoke({"city": city})
    return {"answer": result}

def rag_expert(state: State):
    print("  [知识专家] 处理中...")
    docs = knowledge_retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    response = llm.invoke(f"基于以下资料回答问题：\n{context}\n\n问题：{state['question']}")
    return {"answer": response.content}

def calc_expert(state: State):
    print("  [计算专家] 处理中...")
    import re
    match = re.search(r'[\d+\-*/().\s]+', state["question"])
    if match:
        try:
            result = f"计算结果是：{eval(match.group().strip())}"
        except Exception as e:
            result = f"计算错误：{e}"
    else:
        result = "没有识别到算式。"
    return {"answer": result}

def other_expert(state: State):
    print("  [闲聊专家] 处理中...")
    memory = state.get("user_memory", "") 
    print(f"  [DEBUG] 闲聊专家拿到的记忆: {memory}") # 重点：打印出来看！
    
    prompt = f"{memory}\n用户说：{state['question']}"
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# ========== 3. 定义主管节点 ==========
def supervisor_node(state: State):
    print("  [主管] 正在决定路由...")
    chain = supervisor_prompt | llm
    res = chain.invoke({"question": state["question"]})
    route = res.content.strip().lower()
    
    # 兜底逻辑（防止大模型输出多余废话）
    if "weather" in route:
        route = "weather_expert"
    elif "rag" in route:
        route = "rag_expert"
    elif "calc" in route:
        route = "calc_expert"
    else:
        route = "other_expert"
        
    print(f"  [主管] 决定交给 -> {route}")
    return {"next_node": route}

def save_memory_node(state: State):
    """对话结束后，自动提取记忆存入长期库"""
    messages = state["messages"]
    if len(messages) >= 2:
        human_msg = [m for m in messages if isinstance(m, HumanMessage)][-1].content
        ai_msg = messages[-1].content if messages[-1].content else ""
        
        if ai_msg:
            prompt = f"从对话提取用户个人信息，没有则回'无'。\n用户：{human_msg}\nAI：{ai_msg}"
            res = llm.invoke(prompt).content.strip()
            if res and res != "无":
                memory_store.add_texts([res])
                print(f"[存储长期记忆] {res}")
    return {}


def should_continue(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "save"