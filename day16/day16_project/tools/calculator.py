from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算数学."""
    try:
        return f"结果是：{eval(expression)}"
    except Exception as e:
        return f"错误：{e}"