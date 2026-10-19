import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# ========== 1. 准备数据 ==========
def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

pdf_path = r"Artificial Intelligence.pdf"
full_text = parse_pdf(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
chunks = text_splitter.split_text(full_text)
print(f"切片完成：{len(chunks)} 段")

# ========== 2. 向量检索 ==========
embeddings = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="hybrid_test"
)

# ========== 3. 关键词检索（BM25）==========
tokenized_chunks = [list(chunk) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

def keyword_search(query, k=3):
    """BM25 关键词检索"""
    tokenized_query = list(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices]

# ========== 4. 混合检索 ==========
def hybrid_search(query, k=3):
    """向量 + 关键词，合并去重"""
    vector_results = vectorstore.similarity_search(query, k=k)
    vector_docs = [doc.page_content for doc in vector_results]
    
    keyword_docs = keyword_search(query, k=k)
    
    combined = list(dict.fromkeys(vector_docs + keyword_docs))
    return combined[:k]

# ========== 5. 测试 ==========
query = "教学机智在感知层面有什么新特征？"

print("\n=== 向量检索 ===")
for doc in vectorstore.similarity_search(query, k=3):
    print(f"- {doc.page_content[:80]}...")

print("\n=== 关键词检索 ===")
for doc in keyword_search(query, k=3):
    print(f"- {doc[:80]}...")

print("\n=== 混合检索 ===")
for doc in hybrid_search(query, k=3):
    print(f"- {doc[:80]}...")