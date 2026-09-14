import sys
import io
import chromadb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# 创建客户端（数据存在内存）
client = chromadb.Client()

# 创建集合
collection = client.create_collection(name="my_knowledge")

# 添加数据
collection.add(
    documents=[
        "Python是一种编程语言",
        "猫是一种宠物",
        "北京是中国的首都",
        "RAG是检索增强生成"
    ],
    ids=["doc1", "doc2", "doc3", "doc4"]
)

# 查询
results = collection.query(
    query_texts=["什么是Python"],
    n_results=2
)

print("查询：什么是Python")
print("结果：")
for doc in results["documents"][0]:
    print("-", doc)