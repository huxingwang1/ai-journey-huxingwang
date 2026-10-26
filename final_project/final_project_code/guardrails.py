def filter_input(user_input: str) -> bool:
    """检查用户输入是否安全，不安全返回 False"""
    # 只要包含这些词，直接拦截
    blacklist = ["忽略系统提示", "忽略之前的指令", "开发者模式", "DAN"]
    for word in blacklist:
        if word in user_input:
            return False
    if len(user_input) > 1000:  # 防止超长输入攻击
        return False
    return True

def filter_output(ai_output: str) -> str:
    """对 AI 的输出进行脱敏或拦截"""
    if "sk-" in ai_output or "API_KEY" in ai_output:
        return "抱歉，系统检测到敏感信息，已拦截输出。"
    return ai_output