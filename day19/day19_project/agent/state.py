from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    question: str
    next_node: str
    answer: str
    user_memory: str 