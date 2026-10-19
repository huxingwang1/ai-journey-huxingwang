import sys
import io
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# ========== 1. 初始化基础组件 ==========
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

# 使用之前的向量库（保证有数据）
vectorstore = Chroma(collection_name="hybrid_test", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识库助手。只能根据提供的资料回答问题，不要编造。"),
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

# ========== 2. 初始化缓存向量库 ==========
cache_store = Chroma(collection_name="qa_cache", embedding_function=embeddings)

def ask_with_cache(question):
    start_time = time.time()
    
    # 1. 先去找缓存
    cache_results = cache_store.similarity_search_with_score(question, k=1)
    
    if cache_results:
        doc, score = cache_results[0]
        # 智谱 embedding 距离一般很小，这里放宽阈值到 0.3
        if score < 0.3: 
            print(f"[命中缓存] (距离: {score:.4f}) 直接返回历史答案。耗时: {time.time()-start_time:.2f}秒")
            return doc.metadata["answer"]
    
    # 2. 没命中，真实调用 RAG
    print(f"[未命中缓存] (最佳距离: {score:.4f} > 0.3) 调用 RAG 生成回答...")
    answer = rag_chain.invoke(question)
    
    # 3. 把新的问答存入缓存
    cache_store.add_texts(
        texts=[question],
        metadatas=[{"answer": answer}]
    )
    
    print(f"RAG生成完毕，耗时: {time.time()-start_time:.2f}秒")
    return answer

# ========== 3. 测试 ==========
if __name__ == "__main__":
    print("="*40)
    print("第1次问：")
    ans1 = ask_with_cache("教学机智在感知层面的新特征是什么？")
    print(f"回答：{ans1[:50]}...\n")
    
    print("="*40)
    print("第2次问（换个说法）：")
    ans2 = ask_with_cache("关于教学机智，感知层面有什么新特点？")
    print(f"回答：{ans2[:50]}...\n")
    
    print("="*40)
    print("第3次问（完全相同）：")
    ans3 = ask_with_cache("教学机智在感知层面的新特征是什么？")
    print(f"回答：{ans3[:50]}...")
