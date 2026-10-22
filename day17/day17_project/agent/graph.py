from langgraph.graph import StateGraph, END
from agent.state import State
from agent.nodes import (
    supervisor_node, weather_expert, rag_expert, calc_expert, other_expert
)

# 1. 初始化图
workflow = StateGraph(State)

# 2. 添加所有节点
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("weather_expert", weather_expert)
workflow.add_node("rag_expert", rag_expert)
workflow.add_node("calc_expert", calc_expert)
workflow.add_node("other_expert", other_expert)

# 3. 设置入口：先找主管
workflow.set_entry_point("supervisor")

# 4. 主管做完决定后，根据 next_node 进行条件路由
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    {
        "weather_expert": "weather_expert",
        "rag_expert": "rag_expert",
        "calc_expert": "calc_expert",
        "other_expert": "other_expert"
    }
)

# 5. 所有专家处理完，直接结束
workflow.add_edge("weather_expert", END)
workflow.add_edge("rag_expert", END)
workflow.add_edge("calc_expert", END)
workflow.add_edge("other_expert", END)

# 6. 编译
agent_app = workflow.compile()