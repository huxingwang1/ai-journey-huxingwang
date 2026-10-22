from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    question: str
    next_node: str  # 主管决定去哪个专家
    answer: str     # 专家的最终回答
