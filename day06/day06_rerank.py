import sys
import io
import os
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# 智谱客户端（支持 Rerank）
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 测试 Rerank
query = "年假有多少天？"
documents = [
    "公司年假政策：入职满1年享受5天年假，满3年10天，满5年15天。",
    "报销流程：填写报销单，附上发票，提交给直属领导审批。",
    "考勤制度：上班时间9:00-18:00，迟到超过30分钟扣半天工资。",
    "培训福利：员工每年可申请5000元培训基金。"
]
# 调用 Rerank API
import requests

response = requests.post(
    "https://open.bigmodel.cn/api/paas/v4/rerank",
    headers={
        "Authorization": f"Bearer {os.getenv('ZHIPU_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={
        "model": "rerank",
        "query": query,
        "documents": documents
    }
)

result = response.json()
print("Rerank 结果：")
for item in result["results"]:
    print(f"  索引：{item['index']}，分数：{item['relevance_score']:.4f}")