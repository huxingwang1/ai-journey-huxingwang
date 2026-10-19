import sys, io, os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def llm_evaluate(question, answer, reference):
    """用 LLM 评估回答质量"""
    prompt = f"""请评估下面的回答质量。。

问题：{question}
标准答案：{reference}
待评估回答：{answer}

请从 1-5 分打分，并说明理由。只输出 JSON：
{{"score": 分数, "reason": "理由"}}"""
    
    response = llm.invoke(prompt)
    return response.content

# 测试
result = llm_evaluate(
    "教学机智在感知层面有什么新特征？",
    "根据资料，教学机智在感知层面的新特征是具身感知。",
    "根据资料，教学机智在感知层面的新特征是具身感知。"
)
print(result)