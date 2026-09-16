import sys
import io
from pypdf import PdfReader

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

def parse_pdf(file_path):
    """读取 PDF，返回每页的文本"""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        pages.append(text)
        print(f"第{i+1}页，提取{len(text)}个字符.")
    return pages

result=parse_pdf("Artificial Intelligence.pdf")
print(result)