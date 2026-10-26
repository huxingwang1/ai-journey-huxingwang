from langchain_core.tools import tool
from config import knowledge_retriever

@tool
def search_product_manual(question: str) -> str:
    """从产品手册中搜索信息。用于回答产品功能、售后政策等问题。"""
    docs = knowledge_retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])