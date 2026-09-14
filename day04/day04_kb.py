import sys
import io
import chromadb

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

client = chromadb.Client()
collection = client.create_collection(name="company_docs")

# 模拟公司文档
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

# 测试查询
questions = [
    "年假有多少天？",
    "怎么报销？",
    "可以远程办公吗？",
    "培训基金多少钱？",
    "公司有没有食堂？"  # 测试：资料里没有
]

for q in questions:
    results = collection.query(
        query_texts=[q],
        n_results=1
    )
    print(f"问题：{q}")
    print(f"找到：{results['documents'][0][0]}\n")