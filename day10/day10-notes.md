Day 10 笔记

## MCP 协议
- 统一 AI 和外部工具的交互标准
- MCP Server 提供工具，MCP Client 调用
- 工具复用、生态互通

## LangGraph
- 用图定义 AI 工作流
- State：状态结构
- Node：节点（步骤）
- Edge：边（流转）
- StateGraph：状态图

## 条件分支
- add_conditional_edges
- route 函数返回分支名

## 循环
- 通过 add_edge 回到自己
- should_continue 判断是否继续

## ReAct Agent
- bind_tools 绑定工具
- tool_calls 决定调用哪个工具
- add_edge("tools", "think") 形成循环
- should_continue 判断是否继续

## Agent + RAG
- 把 RAG 封装成 @tool
- Agent 自动判断用哪个工具

## 踩的坑
1. langgraph 需要单独安装
2. State 用 TypedDict 定义
3. add_messages 用于消息列表
4. 工具函数要加 @tool 装饰器
5. 智谱余额不足会报 429
6. 文件名拼写要仔细