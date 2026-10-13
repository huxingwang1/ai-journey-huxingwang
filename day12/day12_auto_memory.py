import sys
import io
import os
from dotenv import load_dotenv
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

# 长期记忆库
memory_store = Chroma(
    collection_name="auto_memory",
    embedding_function=embeddings
)

# ========== 2. 记忆操作函数 ==========
def save_memory(text):
    """保存记忆到向量库."""
    memory_store.add_texts([text])
    print(f"[存储记忆] {text}")

def retrieve_memory(query, k=3):
    """从向量库检索相关记忆"""
    results = memory_store.similarity_search(query, k=k)
    if not results:
        return "暂无相关记忆"
    return "\n".join([doc.page_content for doc in results])

def extract_and_save(question, answer):
    """让 AI 自动提取值得记住的信息并保存"""
    prompt = f"""从下面的对话中提取值得长期记住的信息（如姓名、喜好、地址、职业等）。
如果没有值得记住的，回答"无"。
对话：
用户：{question}
AI：{answer}

只输出要记住的信息，不要解释，不要加标点符号之外的内容。"""
    
    response = llm.invoke(prompt)
    memory = response.content.strip()
    
    if memory and memory != "无":
        save_memory(memory)
    else:
        print("[无需存储]")

# ========== 3. 带记忆的对话函数 ==========
def chat_with_memory(question):
    # 1. 检索相关记忆
    memories = retrieve_memory(question)
    print(f"[检索记忆] {memories[:50]}...")
    
    # 2. 拼接记忆 + 问题
    messages = [
        SystemMessage(content=f"你是一个有记忆的AI助手。以下是你记得的关于用户的信息：\n{memories}"),
        HumanMessage(content=question)
    ]
    
    # 3. 生成回答
    response = llm.invoke(messages)
    answer = response.content
    
    # 4. 自动提取并存储新记忆
    extract_and_save(question, answer)
    
    return answer

# ========== 4. 测试 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("第一次对话：告诉 AI 信息")
    print("=" * 50)
    
    questions1 = [
        "我叫张三，今年25岁",
        "我住在杭州，是一名程序员",
        "我喜欢喝咖啡，尤其是拿铁"
    ]
    
    for q in questions1:
        print(f"\n用户：{q}")
        answer = chat_with_memory(q)
        print(f"AI：{answer[:80]}...")
    
    print("\n" + "=" * 50)
    print("第二次对话：问 AI 之前的信息")
    print("=" * 50)
    
    questions2 = [
        "我叫什么名字？",
        "我住在哪里？",
        "我喜欢喝什么？",
        "我做什么工作？"
    ]
    
    for q in questions2:
        print(f"\n用户：{q}")
        answer = chat_with_memory(q)
        print(f"AI：{answer}")