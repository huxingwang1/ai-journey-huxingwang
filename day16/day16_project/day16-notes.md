Day 16 笔记：模块化与 Docker

## 模块化重构
- config.py：集中管理大模型和向量库实例。
- tools/：每个工具单独一个文件。
- agent/：state, nodes, graph 分层。
- main.py：只负责 API 路由。
- 好处：代码清晰，方便多人协作，方便复用。

## Docker 容器化
- 解决“在我电脑上能跑”的问题。
- Dockerfile：FROM -> WORKDIR -> COPY -> RUN -> CMD。
- 构建：`docker build -t 名字 .`
- 运行：`docker run -d -p 8000:8000 --env-file .env 名字`

## 踩的坑
1. Windows 安装 Docker 需要 WSL2 支持。
2. 千万别把 `.env` 复制进镜像，要在 `.dockerignore` 排除，运行时用 `--env-file` 传入。
3. 容器里的 `host` 必须是 `0.0.0.0`，不能是 `127.0.0.1`。