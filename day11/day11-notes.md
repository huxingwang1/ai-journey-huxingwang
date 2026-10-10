# Day 11 笔记

## Supervisor 模式
- 一个主管 Agent 负责路由
- 多个子 Agent 各司其职
- 用 add_conditional_edges 实现路由
- Supervisor 提示词要清晰

## 子 Agent + 工具
- 每个子 Agent 调用对应工具
- 从问题里提取参数（正则/LLM）
- 工具用 .invoke() 调用

## Human-in-the-loop
- interrupt_before：在节点前暂停
- MemorySaver：保存状态
- thread_id：会话 ID
- invoke(None, config)：继续执行

## 踩的坑
1. Supervisor 提示词要清晰
2. interrupt_before 需要 checkpointer
3. 恢复时要用同一个 thread_id
4. 参数提取要处理边界情况