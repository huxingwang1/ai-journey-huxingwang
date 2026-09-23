import sys
import io
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 定义数据结构
class Person(BaseModel):
    name: str = Field(description="人名")
    age: int = Field(description="年龄")
    city: str = Field(description="城市")

# 解析器
parser = JsonOutputParser(pydantic_object=Person)

# 提示词
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个信息提取助手。\n{format_instructions}"),
    ("user", "从下面这句话提取信息：{text}")
])

# 填充格式说明
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# 链
chain = prompt | llm | parser

# 测试
result = chain.invoke({"text": "张三今年25岁，住在杭州。"})
print(result)
print(type(result))
print(result["name"], result["age"], result["city"])