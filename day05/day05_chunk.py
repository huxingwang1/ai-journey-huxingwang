def split_text(text, chunk_size=100, overlap=20):
    """把长文本切成小段."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 重叠部分，避免切断语义
    return chunks

# 测试
text = "公司年假政策：入职满1年享受5天年假，满3年10天，满5年15天。报销流程：填写报销单，附上发票，提交给直属领导审批，财务3个工作日内打款。"

chunks = split_text(text, chunk_size=50, overlap=10)
for i, chunk in enumerate(chunks):
    print(f"片段{i+1}：{chunk}")