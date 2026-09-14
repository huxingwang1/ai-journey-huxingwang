import sys
import io
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def chat(messages):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    return response.choices[0].message.content

def extract_info(text):
    """从文本提取信息，返回字典"""
    messages = [
        {"role": "system", "content": """你是一个信息提取助手。
只输出JSON，不要任何其他文字，不要用markdown代码块。
格式：{"name": "", "age": 0, "city": ""}"""},
        {"role": "user", "content": f"从下面这句话里提取信息：{text}"}
    ]
    result = chat(messages)
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"error": "解析失败", "raw": result}

# 测试
texts = [
    "张三今年25岁，住在杭州。",
    "李四今年30岁，住在北京。",
    "王五今年28岁，住在上海。"
]

results = []
for text in texts:
    print(f"处理：{text}")
    info = extract_info(text)
    print(f"结果：{info}\n")
    results.append(info)

# 保存结果
with open("day03_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("结果已保存到 day03_output.json")