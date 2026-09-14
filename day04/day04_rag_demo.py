import sys
import io
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# 大模型客户端（DeepSeek）
llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Embedding 客户端（智谱）
embed_client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

def get_embedding(text):
    response = embed_client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding

# 用智谱 Embedding 替换默认模型
class ZhipuEmbedding(embedding_functions.EmbeddingFunction):
    def __call__(self, texts):
        return [get_embedding(t) for t in texts]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="company_docs",
    embedding_function=ZhipuEmbedding()
)

documents = [
    "公司年假政策：入职满1年享受5天年假，满3年10天，满5年15天。",
    "报销流程：填写报销单，附上发票，提交给直属领导审批，财务3个工作日内打款。",
    "考勤制度：上班时间9:00-18:00，迟到超过30分钟扣半天工资。",
    "远程办公：每周可申请1天远程办公，需提前1天向领导报备。",
    "培训福利：员工每年可申请5000元培训基金，用于购买课程或参加行业会议。"
]

collection.add(
    documents=documents,
    ids=[f"doc{i}" for i in range(len(documents))]
)

def rag_query(question):
    results = collection.query(
        query_texts=[question],
        n_results=2
    )
    retrieved_docs = results["documents"][0]
    context = "\n".join(retrieved_docs)
    
    messages = [
        {"role": "system", "content": "你是一个公司政策助手。只能根据提供的资料回答问题，不要编造。如果资料里没有，就说'资料中没有相关信息'。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
    ]
    
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    
    return response.choices[0].message.content

questions = [
    "年假有多少天？",
    "怎么报销？",
    "公司有没有健身房？"
]

for q in questions:
    print(f"问题：{q}")
    print(f"回答：{rag_query(q)}\n")