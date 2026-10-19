import sys
import io
import os
import time
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# ========== 1. 初始化 RAG 链 ==========
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

vectorstore = Chroma(collection_name="hybrid_test", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识库助手。只能根据提供的资料回答问题，不要编造。请尽量简短回答。"),
    ("user", "参考资料：\n{context}\n\n问题：{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 测试用的三个问题
questions = [
    "教学机智在感知层面有什么新特征？",
    "教师如何保持教学机智？",
    "人工智能对教育有什么影响？"
]

# ========== 2. 同步执行 (逐个等待) ==========
def run_sync():
    print("\n" + "="*20 + " 开始同步处理 " + "="*20)
    start_time = time.time()
    
    results = []
    for q in questions:
        print(f"正在处理：{q} ...")
        # 用 invoke 同步调用
        result = rag_chain.invoke(q)
        results.append(result)
        
    end_time = time.time()
    print(f"同步处理完毕，耗时：{end_time - start_time:.2f} 秒")
    return results

# ========== 3. 异步并发 (同时处理) ==========
async def async_rag(question):
    """单个异步任务"""
    # 用 ainvoke 异步调用
    return await rag_chain.ainvoke(question)

async def run_async():
    print("\n" + "="*20 + " 开始异步并发处理。 " + "="*20)
    start_time = time.time()
    
    # 生成任务列表
    tasks = [async_rag(q) for q in questions]
    
    # 用 gather 并发执行
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"异步并发处理完毕，耗时：{end_time - start_time:.2f} 秒")
    return results

# ========== 4. 主函数 ==========
if __name__ == "__main__":
    # 1. 跑同步
    results_sync = run_sync()
    
    # 稍微停一下，防止触发 API 频控
    time.sleep(2)
    
    # 2. 跑异步
    results_async = asyncio.run(run_async())
    
    print("\n" + "="*20 + " 最终总结 " + "="*20)
    print("你会发现：异步并发几乎是同步耗时的 1/3 (因为3个任务同时跑)。")