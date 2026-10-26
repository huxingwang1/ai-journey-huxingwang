from langchain_core.tools import tool

@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态。订单号格式如 ORD001。"""
    # 模拟真实的数据库查询
    mock_orders = {
        "ORD001": "已发货，预计明天送达。",
        "ORD002": "待付款，请尽快支付。",
        "ORD003": "已完成，感谢您的购买。"
    }
    return mock_orders.get(order_id, f"未查询到订单 {order_id} 的信息。")