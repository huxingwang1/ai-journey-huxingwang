from agent.graph import agent_app

questions = [
    "你好，我叫李雷",                      # 测记忆 + 闲聊
    "查一下我的订单 ORD001",                # 测订单专家
    "你们公司的退货政策是什么？",            # 测 RAG 专家
    "我要投诉你们！给我转人工！"            # 测转人工
]

for q in questions:
    print(f"\n{'='*40}")
    print(f"用户：{q}")
    result = agent_app.invoke({"question": q})
    print(f"最终回答：{result['answer']}")
    if result.get("need_human"):
        print("⚠️ 触发转人工！")