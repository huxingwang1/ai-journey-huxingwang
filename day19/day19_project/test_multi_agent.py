from agent.graph import agent_app

questions = [
    "北京今天天气怎么样？",
    "教学机智在感知层面有什么新特征？",
    "计算 123 * 456",
    "你好，你是谁？"
]

for q in questions:
    print(f"\n{'='*40}")
    print(f"用户：{q}")
    result = agent_app.invoke({"question": q})
    print(f"最终回答：{result['answer'][:100]}...")
