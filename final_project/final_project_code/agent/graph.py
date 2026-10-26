from langgraph.graph import StateGraph, END
from agent.state import State
from agent.nodes import (
    supervisor_node, order_expert, rag_expert, human_expert, other_expert, load_memory_node
)

workflow = StateGraph(State)

# 1. 添加所有节点
workflow.add_node("memory", load_memory_node)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("order_expert", order_expert)
workflow.add_node("rag_expert", rag_expert)
workflow.add_node("human_expert", human_expert)
workflow.add_node("other_expert", other_expert)

# 2. 入口设为记忆节点
workflow.set_entry_point("memory")
workflow.add_edge("memory", "supervisor")

# 3. 主管路由（条件边）
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    {
        "order_expert": "order_expert",
        "rag_expert": "rag_expert",
        "human_expert": "human_expert",
        "other_expert": "other_expert"
    }
)

# 4. 专家处理完后全部结束
workflow.add_edge("order_expert", END)
workflow.add_edge("rag_expert", END)
workflow.add_edge("human_expert", END)
workflow.add_edge("other_expert", END)

# 5. 编译
agent_app = workflow.compile()