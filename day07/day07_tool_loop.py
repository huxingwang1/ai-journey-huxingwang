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

# ========== 定义工具 ==========
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
    """计算数学表达式，如 '123 * 456'"""
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
            "description": "计算数学表达式",
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

def chat_with_tools_full(question):
    messages = [
        {"role": "system", "content": "你是一个助手，可以调用工具回答问题。"},
        {"role": "user", "content": question}
    ]
    
    while True:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # 如果没有工具调用，结束
        if not message.tool_calls:
            return message.content
        
        # 执行每个工具调用
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            print(f"  [调用] {tool_name}({tool_args})")
            
            if tool_name == "get_weather":
                result = get_weather(**tool_args)
            elif tool_name == "calculator":
                result = calculator(**tool_args)
            else:
                result = "未知工具"
            
            # 把结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

# 测试
questions = [
    "北京今天天气怎么样？",
    "帮我算一下 123 * 456",
    "杭州天气怎么样？如果下雨提醒我带伞"
]

for q in questions:
    print(f"\n{'='*40}")
    print(f"问题：{q}")
    answer = chat_with_tools_full(q)
    print(f"回答：{answer}")