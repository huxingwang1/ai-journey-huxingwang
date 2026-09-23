from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    count: int
    result: str

def increment_node(state: State):
    count = state["count"] + 1
    print(f"[循环] 第 {count} 次")
    return {"count": count}

def should_continue(state: State):
    if state["count"] < 3:
        return "continue"
    return "end"

workflow = StateGraph(State)
workflow.add_node("increment", increment_node)
workflow.set_entry_point("increment")
workflow.add_conditional_edges(
    "increment",
    should_continue,
    {
        "continue": "increment",   # ← 回到自己，形成循环
        "end": END
    }
)

app = workflow.compile()
result = app.invoke({"count": 0})
print(f"最终 count：{result['count']}")