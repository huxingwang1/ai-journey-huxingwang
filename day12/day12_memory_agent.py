import sys
import io
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# ========== 1. 初始化 ==========
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

memory_store = Chroma(
    collection_name="agent_memory",
    embedding_function=embeddings
)

# ========== 2. 状态 ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ========== 3. 节点1：检索记忆 + 回答 ==========
def memory_chat_node(state: State):
    """检索长期记忆，生成回答"""
    last_message = state["messages"][-1]
    question = last_message.content
    
    # 检索相关记忆
    memories = memory_store.similarity_search(question, k=3)
    memory_text = "\n".join([doc.page_content for doc in memories]) if memories else "暂无相关记忆"
    
    # 拼接记忆 + 问题
    messages = [
        SystemMessage(content=f"你是一个有记忆的AI助手。以下是你记得的关于用户的信息：\n{memory_text}"),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# ========== 4. 节点2：提取 + 存储记忆 ==========
def save_memory_node(state: State):
    """从对话中提取值得记住的信息并存储"""
    messages = state["messages"]
    
    if len(messages) >= 2:
        question = messages[-2].content
        answer = messages[-1].content
        
        prompt = f"""从下面的对话中提取值得长期记住的信息（如姓名、喜好、地址、职业等）。
如果没有值得记住的，回答"无"。
对话：
用户：{question}
AI：{answer}

只输出要记住的信息，不要解释。"""
        
        response = llm.invoke(prompt)
        memory = response.content.strip()
        
        if memory and memory != "无":
            memory_store.add_texts([memory])
            print(f"[存储记忆] {memory}")
        else:
            print("[无需存储]")
    
    return {}

# ========== 5. 构建图 ==========
workflow = StateGraph(State)
workflow.add_node("chat", memory_chat_node)
workflow.add_node("save", save_memory_node)

workflow.set_entry_point("chat")
workflow.add_edge("chat", "save")
workflow.add_edge("save", END)

app = workflow.compile()

# ========== 6. 测试 ==========
if __name__ == "__main__":
    questions = [
        "我叫张三，今年25岁",
        "我住在杭州，是一名程序员",
        "我喜欢喝咖啡",
        "我叫什么名字？",
        "我住在哪里？",
        "我喜欢什么？"
    ]
    
    for q in questions:
        print(f"\n{'='*40}")
        print(f"用户：{q}")
        result = app.invoke({"messages": [HumanMessage(content=q)]})
        print(f"AI：{result['messages'][-1].content}")