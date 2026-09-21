Day 6 笔记

## FastAPI
- @app.get / @app.post 定义接口
- Pydantic 定义请求/响应模型
- uvicorn 启动服务
- /docs 自动生成文档

## RAG API
- 启动时构建知识库
- /query 接口：检索 + Rerank + 生成
- 返回 answer + sources

## Rerank
- 先检索 10 条候选
- Rerank 精排，取前 3 条
- 智谱 rerank API

## 客户端
- requests.post 调用
- 两个终端：一个跑服务，一个跑客户端

## 踩的坑
1. JSON 引号必须是英文
2. PDF 路径用绝对路径
3. 文件名不要有空格
4. 缩进必须统一
5. import requests 要在顶部
6. --reload 会监听所有文件