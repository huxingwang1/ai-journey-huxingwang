Day 17 笔记：多智能体协作与 Docker Compose

## 一、为什么要用多智能体（Supervisor 模式）
- **痛点**：单智能体（Day 16）绑定了所有工具（天气、计算、知识库）。当工具越来越多时，Prompt 会变得极其臃肿，大模型容易选错工具。
- **解决方案**：引入“主管（Supervisor）+ 专家（Expert）”架构。
- **主管**：只看问题，负责将任务分发给对应的专家。不负责具体解答。
- **专家**：只负责自己领域的任务（如天气专家只查天气，知识专家只查 RAG）。

## 二、代码实现核心点
1. **State 的简化**：不需要存整段对话历史（`messages`），只需要 `question`（用户问题）、`next_node`（主管派给谁）、`answer`（专家的回答）。
2. **条件路由（Conditional Edges）**：
   ```python
   workflow.add_conditional_edges(
       "supervisor",
       lambda state: state["next_node"],
       {
           "weather_expert": "weather_expert",
           "rag_expert": "rag_expert",
           "calc_expert": "calc_expert",
           "other_expert": "other_expert"
       }
   )
专家节点：可以直接调大模型，也可以调用之前封装好的 @tool 工具。

兜底逻辑：主管大模型的输出可能带有废话（比如“我觉得应该给天气专家”），在 supervisor_node 里要用 if "weather" in route 等逻辑做规范化，防止找不到节点报错。

三、Docker Compose
作用：一条命令（docker-compose up -d）同时启动多个容器（API + Redis）。

配置文件：docker-compose.yml。

服务名通信：容器之间可以用服务名通信（如代码里写 REDIS_HOST=redis），不需要写 IP。

解决网络问题：如果在容器内下载依赖失败（如 tiktoken），要在 docker-compose.yml 的 ai-api 服务下加上国内 DNS：

yaml
dns:
  - 114.114.114.114
  - 8.8.8.8
四、踩的坑
空文件报错：docker-compose.yml 修改后必须 Ctrl + S 保存，否则会报 empty compose file。

端口冲突：如果之前用 docker run 启动了容器，或者本地开了 uvicorn，再次跑 Compose 会报 port is already allocated。需先清理旧容器。

跨文件导入：nodes.py 里如果没有导入 knowledge_retriever，会引发 Pylance 警告或运行报错。

依赖下载失败：容器内 DNS 解析失败，加上 dns 配置即可。