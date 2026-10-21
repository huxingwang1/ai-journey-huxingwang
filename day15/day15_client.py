import requests

url = "http://127.0.0.1:8000/chat"
session_id = "user_001"

# 第 1 轮
print("第 1 轮：")
res1 = requests.post(url, json={"message": "我叫张三，喜欢喝咖啡", "session_id": session_id})
print(f"AI：{res1.json()['reply']}\n")

# 第 2 轮
print("第 2 轮：")
res2 = requests.post(url, json={"message": "今天北京天气怎么样？", "session_id": session_id})
print(f"AI：{res2.json()['reply']}\n")

# 第 3 轮（测试长期记忆是否生效）
print("第 3 轮：")
res3 = requests.post(url, json={"message": "我叫什么名字？喜欢喝什么？", "session_id": session_id})
print(f"AI：{res3.json()['reply']}")