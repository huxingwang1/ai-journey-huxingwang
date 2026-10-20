
import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from day14_stream import vectorstore



load_dotenv()

# 初始化模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 初始化 Embedding 和 向量库
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
vectorstore = Chroma(collection_name="hybrid_test", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# 定义压缩提示词
compress_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个信息提取助手。请从下面的文档中，只提取与问题相关的句子。如果完全不相关，输出'无'。"),
    ("user", "问题：{question}\n\n文档：{document}")
])

compress_chain = compress_prompt | llm | StrOutputParser()

def compress_docs_batch(question, docs):
    """用 batch 并发压缩"""
    inputs = [{"question": question, "document": doc.page_content} for doc in docs]
    
    # 核心：用 batch() 代替循环 invoke()
    results = compress_chain.batch(inputs)
    
    compressed_docs = []
    for result in results:
        if result.strip() and result.strip() != "无":
            compressed_docs.append(result)
    return compressed_docs
# 测试
query = "教学机智在感知层面有什么新特征？"
raw_docs = retriever.invoke(query) # 这里返回的是 Document 对象列表
print(f"压缩前条数：{len(raw_docs)}")

compressed = compress_docs_batch(query, raw_docs)
print(f"压缩后条数：{len(compressed)}")
print(f"压缩后内容：\n{compressed[0]}")