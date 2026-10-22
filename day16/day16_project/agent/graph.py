from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import State
from agent.nodes import memory_retrieval_node, think_node, tool_node, save_memory_node, should_continue

workflow = StateGraph(State)
workflow.add_node("memory", memory_retrieval_node)
workflow.add_node("think", think_node)
workflow.add_node("tools", tool_node)
workflow.add_node("save", save_memory_node)

workflow.set_entry_point("memory")
workflow.add_edge("memory", "think")
workflow.add_conditional_edges("think", should_continue, {"tools": "tools", "save": "save"})
workflow.add_edge("tools", "think")
workflow.add_edge("save", END)

memory = MemorySaver()
agent_app = workflow.compile(checkpointer=memory)