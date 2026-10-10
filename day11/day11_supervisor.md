Supervisor 模式 = 一个“主管”Agent 负责调度多个“子”Agent。

为什么要多 Agent？

单个 Agent 工具太多时，容易选错。

不同领域交给不同专家，更准确。

更贴近真实团队协作。

架构：

text
用户输入
   ↓
┌──────────────┐
│  Supervisor  │  ← 判断问题类型，路由到子 Agent
└──────┬───────┘
       ↓
 ┌─────┼─────┬─────┐
 ↓     ↓     ↓     ↓
知识  订单  天气  计算
Agent Agent Agent Agent