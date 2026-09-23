import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 链1：翻译成英文
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "把用户的话翻译成英文。"),
    ("user", "{text}")
])
translate_chain = translate_prompt | llm | StrOutputParser()

# 链2：总结
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "用一句话总结用户的话。"),
    ("user", "{text}")
])
summary_chain = summary_prompt | llm | StrOutputParser()

# 并行执行两条链
parallel_chain = RunnableParallel(
    translation=translate_chain,
    summary=summary_chain
)

# 测试
result = parallel_chain.invoke({"text": "人工智能正在改变教育的方式。"})
print("翻译：", result["translation"])
print("总结：", result["summary"])