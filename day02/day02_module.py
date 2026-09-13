# 导入标准库
import os
import json
import random
from datetime import datetime

# os：操作系统相关
print("当前目录：", os.getcwd())
print("文件列表：", os.listdir("."))

# json：处理JSON数据（后面API全靠它）
data = {"name": "张三", "age": 25, "skills": ["Python", "AI"]}
json_str = json.dumps(data, ensure_ascii=False)
print("JSON字符串：", json_str)

parsed = json.loads(json_str)
print("解析回来：", parsed)

# datetime：时间
now = datetime.now()
print("当前时间：", now.strftime("%Y-%m-%d %H:%M:%S"))

# random：随机
print("随机数：", random.randint(1, 100))