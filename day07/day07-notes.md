#Day 7 笔记

## Agent 基础
- 感知、规划、行动、记忆
- ReAct：Thought → Action → Observation

## Tool Calling
- 定义工具（函数 + 描述）
- 传给 LLM
- LLM 返回 tool_calls
- 执行工具，返回结果
- 结果加进 messages（role="tool"）
- LLM 生成最终回答

## 工具定义
- name：函数名
- description：告诉 AI 什么时候用
- parameters：参数结构（JSON Schema）

## 关键点
- AI 只"决定"调用，不执行
- tool_call_id 必须匹配
- max_iterations 防止死循环

## 踩的坑
1. 工具描述要清晰
2. tool_call_id 要匹配
3. json.loads 解析参数
4. 消息历史要完整保存