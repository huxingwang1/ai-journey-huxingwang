from langchain_core.tools import tool
from config import knowledge_retriever

@tool
def search_knowledge(question: str) -> str:
    """从知识库搜索信息"""
    docs = knowledge_retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])