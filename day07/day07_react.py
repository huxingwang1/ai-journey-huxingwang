"""
Day 7: ReAct Agent
让 AI 显式地"思考"和"行动"，循环调用工具直到完成任务。
"""

import sys
import io
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# 解决 Windows 终端中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

# DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ========== 定义工具函数 ==========
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    weather_data = {
        "北京": "晴，25°C，湿度40%",
        "上海": "多云，28°C，湿度65%",
        "杭州": "小雨，22°C，湿度80%",
        "深圳": "晴，30°C，湿度70%"
    }
    return weather_data.get(city, f"暂时没有{city}的天气数据")

def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{e}"

# ========== 工具定义（给 AI 看的）==========
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，如 '123 * 456'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 123 * 456"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# ========== ReAct Agent ==========
def react_agent(question, max_iterations=5):
    """ReAct Agent：思考 → 行动 → 观察 → 循环"""
    messages = [
        {
            "role": "system",
            "content": "你是一个能调用工具的 Agent。每次回答前，先思考需要什么信息，然后调用工具获取。"
        },
        {"role": "user", "content": question}
    ]
    
    for i in range(max_iterations):
        print(f"\n--- 第 {i+1} 轮 ---")
        
        # 调用 AI
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # 如果 AI 没有调用工具，说明它准备好回答了
        if not message.tool_calls:
            print(f"[最终回答] {message.content}")
            return message.content
        
        # 执行工具
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"[思考] 我需要调用 {tool_name}")
            print(f"[行动] {tool_name}({tool_args})")
            
            # 执行对应的工具
            if tool_name == "get_weather":
                result = get_weather(**tool_args)
            elif tool_name == "calculator":
                result = calculator(**tool_args)
            else:
                result = "未知工具"
            
            print(f"[观察] {result}")
            
            # 把结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    
    return "达到最大迭代次数"

# ========== 测试 ==========
if __name__ == "__main__":
    questions = [
        "北京天气怎么样？",
        "杭州天气怎么样？如果下雨提醒我带伞",
        "计算 (123 + 456) * 2"
    ]
    
    for q in questions:
        print(f"\n{'='*50}")
        print(f"问题：{q}")
        react_agent(q)