# interview-agent
Agent that takes a JD as input and outputs a tailored resume, company research, and interview questions using multi-agent orchestration, RAG, memory, reflection, and human-in-the-loop.

## 架构设计

### 整体模式：中心化编排（Orchestrator Pattern）

`main.py` 作为唯一编排器，负责调度所有 agent，agent 之间不直接通信，均通过编排器传递数据。

### 执行流程

```
用户输入 JD
     │
     ▼
jd_parser                     串行 — 结构化提取 company / role / role_type / key_skills
     │
     ▼
Memory check                  查历史投递记录（find_similar）
     │
     ├─────────────────────────────────┐
     ▼                                 ▼
resume_agent                    research_agent        Phase 1：并行
RAG 召回 + 简历改写 + Reflection  Tavily 搜索 + 汇总
     │                                 │
     └───────────────┬─────────────────┘
                     ▼
            Human-in-the-loop          用户确认/补充公司研究结果
                     │
                     ▼
            interview_agent            Phase 2：串行 — 依赖 research_result 生成面试题库
                     │
                     ▼
            输出报告（outputs/）+ 写入 memory/history.json
```

### 关键设计决策

| 设计点 | 实现方式 | 原因 |
|---|---|---|
| Phase 1 并行 | `asyncio + ThreadPoolExecutor(max_workers=2)` | `resume_agent` 与 `research_agent` 互不依赖，并行节省等待时间 |
| Phase 2 串行 | 等待 Phase 1 完成后调用 | `interview_agent` 需要公司研究结果才能生成有针对性的题目 |
| Human-in-the-loop | Phase 1 和 Phase 2 之间暂停 | 让用户校验/补充公司信息，确保 interview agent 拿到准确上下文 |
| 数据传递 | 编排器显式传参（`jd_info` dict + 结果字符串） | 避免 agent 间隐式耦合，数据流清晰可追踪 |

### Agent 职责

- **jd_parser** — LLM 结构化提取 JD 关键信息
- **resume_agent** — RAG 召回个人经历 + LLM 改写简历 + Reflection 自我审查
- **research_agent** — Tavily 搜索（3 queries × max 3 results）+ LLM 汇总公司信息
- **interview_agent** — 基于 JD + 公司研究生成定制面试题库
