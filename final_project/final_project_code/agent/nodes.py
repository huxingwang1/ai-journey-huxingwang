from config import llm, knowledge_retriever, redis_client
from agent.state import State
from langchain_core.prompts import ChatPromptTemplate
from tools.order import query_order
from tools.rag import search_product_manual
import re

# ========== 1. 主管提示词（极高严格度） ==========
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个企业级客服主管。请判断用户意图，并严格输出以下专家代号之一：
- order_expert：用户需要查询订单、物流，或提到了“订单号”、“发货”、“买了什么”。
- rag_expert：用户询问产品功能、售后政策、公司规定、退换货流程等需要查资料的问题。
- human_expert：用户情绪激动，提到“投诉”、“退款”、“人工”、“找你们领导”。
- other_expert：纯粹的打招呼、闲聊（如“你好”、“你是谁”）。

【绝对规则】：你只能输出上述四个英文代号中的一个！绝对不要输出任何标点、解释或多余的字！"""),
    ("user", "{question}")
])

# ========== 2. 专家节点 ==========
def order_expert(state: State):
    print("  [订单专家] 处理中...")
    # 正则提取订单号
    match = re.search(r'ORD\d+', state["question"])
    if match:
        order_id = match.group()
        result = query_order.invoke({"order_id": order_id})
    else:
        result = "系统未识别到有效的订单号，请提供如 ORD001 格式的编号，以便我为您查询。"
    return {"answer": result}

def rag_expert(state: State):
    print("  [RAG 专家] 处理中...")
    # 调用工具查库
    context = search_product_manual.invoke({"question": state["question"]})
    
    # 结合上下文让大模型总结
    response = llm.invoke(f"你是一个客服专家。请基于以下公司资料回答用户问题。如果资料中没有提及，请直接说“抱歉，我暂时没有查到相关规定”。\n\n资料：\n{context}\n\n问题：{state['question']}")
    return {"answer": response.content}

def human_expert(state: State):
    print("  [转人工] 触发人工介入...")
    return {"answer": "已经为您转接人工客服，客服代表将很快介入，请稍候。", "need_human": True}

def other_expert(state: State):
    print("  [闲聊专家] 处理中...")
    memory = state.get("user_memory", "")
    response = llm.invoke(f"{memory}\n用户说：{state['question']}")
    return {"answer": response.content}

# ========== 3. 路由主管节点 ==========
def supervisor_node(state: State):
    print("  [主管] 正在决定路由...")
    chain = supervisor_prompt | llm
    res = chain.invoke({"question": state["question"]})
    route = res.content.strip().lower()
    
    # 兜底清洗
    if "order" in route:
        route = "order_expert"
    elif "rag" in route:
        route = "rag_expert"
    elif "human" in route:
        route = "human_expert"
    else:
        route = "other_expert"
        
    print(f"  [主管] 决定交给 -> {route}")
    return {"next_node": route}

# ========== 4. 记忆节点 ==========
def load_memory_node(state: State):
    """从 Redis 读取并保存用户信息"""
    question = state["question"]
    memory_str = ""
    
    # 存名字
    if "我叫" in question and not any(q in question for q in ["什么", "啥", "谁"]):
        match = re.search(r'我叫([\u4e00-\u9fa5]{1,3})', question)
        if match:
            name = match.group(1)
            redis_client.set("user:name", name)
            print(f"[Redis] 已保存用户名：{name}")
    
    # 读名字
    name = redis_client.get("user:name")
    if name:
        print(f"[Redis] 命中记忆：{name}")
        memory_str = f"用户的真实名字叫：{name}"
            
    return {"user_memory": memory_str}
