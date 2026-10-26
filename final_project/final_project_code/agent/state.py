from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    question: str
    next_node: str
    answer: str
    user_memory: str  # Redis 长期记忆
    need_human: bool  # 新增：是否需要转人工