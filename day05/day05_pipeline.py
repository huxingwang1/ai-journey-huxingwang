import sys
import io
import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# 智谱 Embedding 客户端
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

class ZhipuEmbedding(embedding_functions.EmbeddingFunction):
    def __call__(self, texts):
        return [get_embedding(t) for t in texts]

def parse_pdf(file_path):
    """读取 PDF，返回所有页的文本"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=200, overlap=50):
    """切片"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 主流程
print("1. 解析 PDF...")
text = parse_pdf("Artificial Intelligence.pdf")
print(f"   提取了 {len(text)} 个字符")

print("2. 切片...")
chunks = split_text(text)
print(f"   切成 {len(chunks)} 段")

print("3. 存入 Chroma...")
client = chromadb.Client()
collection = client.create_collection(
    name="pdf_knowledge",
    embedding_function=ZhipuEmbedding()
)

collection.add(
    documents=chunks,
    ids=[f"chunk{i}" for i in range(len(chunks))]
)
print(f"   已存入 {len(chunks)} 段")

print("\n4. 测试查询...")
questions = [
    "人工智能时代，教学机智在“感知、判断、行动”三个层面分别出现了哪些新特征？",
    "教师—技术—情境”三元关系中，教师如何保持教学机智而不被技术支配？",
    "智能技术进入课堂，是增强还是削弱了教师的教学机智？"
]

for q in questions:
    results = collection.query(
        query_texts=[q],
        n_results=1
    )
    print(f"\n问题：{q}")
    print(f"找到：{results['documents'][0][0][:100]}...")

llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def rag_answer(question, collection, n_results=3):
    """RAG 完整流程：检索 + 生成"""
    # 1. 检索
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    retrieved_docs = results["documents"][0]
    context = "\n\n".join(retrieved_docs)
    
    # 2. 生成
    messages = [
        {"role": "system", "content": "你是一个知识库助手。只能根据提供的资料回答问题，不要编造。如果资料里没有，就说'资料中没有相关信息'。"},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
    ]
    
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    
    return response.choices[0].message.content

# 测试完整 RAG
print("\n" + "=" * 40)
print("完整 RAG 问答.：")
for q in questions:
    print(f"\n问题：{q}")
    print(f"回答：{rag_answer(q, collection)}")