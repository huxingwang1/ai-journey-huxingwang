from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询天气"""
    return f"{city}今天晴，25°C。"