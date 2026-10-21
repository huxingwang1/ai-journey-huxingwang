综合 Agent 架构设计

## 核心组件
1. LLM：DeepSeek
2. 路由：Supervisor 模式（判断用户意图）
3. 工具：get_weather, calculator, search_knowledge
4. 记忆：LangGraph 的 MemorySaver (短期) + Chroma 向量库。 (长期)

## 数据流
用户输入 -> 提取长期记忆并注入 -> Supervisor 判断意图 -> 路由到对应子 Agent/工具执行 -> 生成回答 -> 存入短期和长期记忆。