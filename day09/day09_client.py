import requests

url = "http://127.0.0.1:8000/query"

questions = [
    "教学机智在感知层面有什么新特征？",
    "教师如何保持教学机智？",
    "公司有没有健身房？"
]

for q in questions:
    response = requests.post(url, json={"question": q})
    data = response.json()
    print(f"问题：{q}")
    print(f"回答：{data['answer'][:150]}...")
    print("-" * 50)