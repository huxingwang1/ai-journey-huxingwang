import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
 
load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 提示词带历史占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的AI助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}")
])

# 基础链
chain = prompt | llm | StrOutputParser()

# 存储会话历史
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 带历史的链
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# 测试
config = {"configurable": {"session_id": "user1"}}

print("第1轮：")
print(chain_with_history.invoke({"question": "我叫张三"}, config=config))

print("\n第2轮：")
print(chain_with_history.invoke({"question": "我叫什么名字？"}, config=config))