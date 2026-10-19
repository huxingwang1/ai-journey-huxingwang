import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 复用 vectorstore
from day13_hybrid import vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识库助手。只能根据提供的资料回答问题，不要编造。如果资料里没有，就说'资料中没有相关信息'。"),
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

# ========== 评估集 ==========
eval_set = [
    {
        "question": "教学机智在感知层面有什么新特征？",
        "expected": "具身感知"
    },
    {
        "question": "教师如何保持教学机智？",
        "expected": "持续反思。"
    },
    {
        "question": "公司有没有健身房？",
        "expected": "资料中没有相关信息"
    }
]

def evaluate(rag_chain, eval_set):
    results = []
    for item in eval_set:
        question = item["question"]
        expected = item["expected"]
        
        answer = rag_chain.invoke(question)
        hit = expected in answer
        
        results.append({
            "question": question,
            "expected": expected,
            "answer": answer[:100],
            "hit": hit
        })
        
        print(f"问题：{question}")
        print(f"期望：{expected}")
        print(f"回答：{answer[:100]}...")
        print(f"命中：{hit}")
        print("-" * 40)
    
    hits = sum(1 for r in results if r["hit"])
    accuracy = hits / len(results)
    print(f"\n准确率：{accuracy:.2%} ({hits}/{len(results)})")
    
    return results

evaluate(rag_chain, eval_set)