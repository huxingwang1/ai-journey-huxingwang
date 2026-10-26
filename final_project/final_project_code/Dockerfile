# 1. 指定基础镜像（Python 3.12 精简版）
FROM python:3.12-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 先复制依赖清单并安装（利用 Docker 缓存加速后续构建）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 复制当前目录下的所有代码到容器的 /app 里
COPY . .

# 5. 暴露端口
EXPOSE 8000

# 6. 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]