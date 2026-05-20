# Job Interview Prep Agent — 设计文档

**日期：** 2026-05-11
**状态：** 待实现

---

## 一、项目概述

一个面向求职者的 AI Agent，输入一条 JD（职位描述），自动完成三件事：

1. 基于个人经历知识库，定制化改写简历 bullets
2. 研究目标公司业务线、产品和文化
3. 生成针对该岗位的面试题库（含回答框架提示）

**核心叙事：** 用这个 agent 准备 AI PM 面试，在面试现场 live demo，同时展示产品思维和技术深度。

---

## 二、技术栈覆盖

| Agent 能力 | 实现方式 |
|---|---|
| Multi-agent 协作 | Orchestrator + 3 个专职 sub-agent |
| Tool Use | Web 搜索（Tavily API）、RAG 查询 |
| RAG / 向量检索 | ChromaDB 本地向量库，存储个人经历文档 |
| 并发规划 | asyncio 两阶段并行调度 |
| Memory（跨会话） | 本地 JSON 存储历史投递记录 |
| Reflection（自我反思） | Resume Agent 内部 self-critique 循环 |
| Human-in-the-loop | Research 完成后暂停，用户确认后继续 |
| 结构化输出 | Markdown 报告，rich 终端渲染 |

---

## 三、整体架构

```
JD 输入
    ↓
Orchestrator
  · 解析 JD：调用 Claude API 提取公司名、岗位类型、核心技能要求（结构化 JSON 输出）
  · 查询 Memory：对比历史投递记录，提示相似申请
    ↓
Phase 1（asyncio 并行）
    ├── Resume Agent
    │     · RAG 召回相关经历片段
    │     · 生成定制 bullet points
    │     · Reflection：self-critique → 改写
    │
    └── Research Agent
          · Tavily 搜索公司信息（3-4 次查询）
          · 汇总业务线、产品、文化、近期动态
    ↓
⏸ Human-in-the-loop
  · 展示 Research Agent 结果
  · 用户确认 / 补充 / 纠正
    ↓
Phase 2
    └── Interview Prep Agent
          · 输入：确认后的公司信息 + JD + 用户背景摘要
          · 生成行为题、产品题、技术认知题、反问题
          · 每题附回答框架提示
    ↓
聚合输出 Markdown 报告
    ↓
写入 Memory（history.json）
```

---

## 四、知识库结构（RAG）

用户需在 `/knowledge` 目录下维护以下文档：

```
knowledge/
├── resume_base.md          # 完整简历底稿（所有经历，不删减）
├── experiences/
│   ├── internship_A.md     # 每段实习详细描述（含数据、背景、故事）
│   └── internship_B.md
├── projects/
│   └── project_X.md        # 项目经历
└── self_profile.md         # 个人标签：擅长方向、核心优势、求职意向
```

**RAG 工作方式：**
- 启动时将所有文档 chunk 并 embed，存入 ChromaDB
- Resume Agent 用 JD 关键词做语义检索，召回最相关的 2-3 个片段
- 同一段经历，投商业化岗突出 GMV / 转化率，投 AI 产品岗突出模型落地 / 数据飞轮

**Memory 格式（`/memory/history.json`）：**

```json
{
  "applications": [
    {
      "date": "2026-05-11",
      "company": "字节跳动",
      "role": "AI产品经理",
      "jd_keywords": ["大模型", "C端产品"],
      "output_path": "outputs/bytedance_20260511.md"
    }
  ]
}
```

---

## 五、Sub-Agent 详细设计

### 5.1 Resume Agent

**输入：** JD 文本 + RAG 召回的经历片段（top-3 语义相似度最高的 chunk，chunk size 500 tokens）

**执行步骤：**
1. 分析 JD，判断岗位类型（商业化 / AI / 增长 / 平台等）
2. 为每段相关经历生成 2-3 条 bullet（STAR 格式，数字优先）
3. **Reflection 轮**：用 Claude 对每条 bullet 做 critique
   - 是否体现 JD 要求的核心能力？
   - 是否有量化数据？
   - 是否有动词开头？
4. 根据 critique 输出最终修订版

**输出示例：**
```
【XX 公司实习 — 改写后（对标：AI产品经理）】
· 主导 XX 功能从 0 到 1 上线，30 天 DAU 提升 23%
· 设计大模型辅助审核方案，人工成本降低 40%
· 协调研发/设计/数据三方，按期交付迭代需求
```

---

### 5.2 Research Agent

**输入：** 公司名 + 岗位名

**工具：** Tavily API

**搜索策略（顺序执行）：**
```
查询 1: "{公司名} 业务线 产品矩阵 2025"
查询 2: "{公司名} {岗位方向} 战略 最新动态"
查询 3: "{公司名} 公司文化 价值观 面试风格"
```

**输出结构：**
- 核心业务线（3-5 条，附简要说明）
- 该部门 / 产品近期动态（最新 1-2 条新闻或发布）
- 公司文化关键词
- What You Should Know（3 条面试加分提示）

---

### 5.3 Interview Prep Agent

**输入：** JD + 确认后的公司信息 + 用户经历摘要（从 `knowledge/self_profile.md` 直接读取，不走 RAG）

**题目分类：**

| 类型 | 数量 | 说明 |
|---|---|---|
| 行为题 | 3 条 | 结合用户经历背景，STAR 格式 |
| 产品题 | 3 条 | 结合公司具体产品，如"如何提升 XX 的次日留存" |
| 技术认知题 | 2 条 | AI PM 特有，如"如何评估大模型效果"、"如何定义 AI 功能的成功指标" |
| 反问题 | 2 条 | 建议向面试官提问的高质量问题 |

每道题附：**回答框架提示**（结构提示，不写完整答案）

---

## 六、错误处理

| 异常情况 | 处理方式 |
|---|---|
| Tavily 搜索失败 | Research Agent 降级为仅基于 JD 分析，继续运行 |
| RAG 召回为空 | Resume Agent 提示"未找到相关经历，请补充知识库" |
| Sub-agent 超时（>45s） | Orchestrator 跳过该模块，其余正常输出，报告中标注"未完成" |
| Memory 文件损坏 | 忽略历史记录，正常运行，写入时覆盖 |

---

## 七、Demo 流程脚本（面试现场）

**总时长约 3 分钟**

```
Step 1 [~15s]
  粘贴 JD → 回车
  终端输出：
    ✦ Orchestrator 解析中...
    → 岗位类型：AI 产品经理 | 公司：XX
    → 检测到历史投递：上次投过字节 AI PM，可参考对比

Step 2 [~25s]
  三个进度条同时出现：
    [Resume Agent   ] ████████░░ 召回 2 段经历，生成 bullets...
    [Research Agent ] ██████░░░░ 搜索公司信息（3/4）...

Step 3 [~10s]
  ⏸ HITL 暂停
    公司研究完成，请确认以下信息：
    [展示业务线摘要]
    直接回车确认，或输入补充信息：_

Step 4 [~20s]
  Interview Prep Agent 启动
    基于确认信息生成题库...

Step 5 [~5s]
  ✅ 报告生成完毕 → outputs/xx_20260511.md
  [rich 渲染完整 Markdown 报告]
```

---

## 八、技术依赖

```
anthropic>=0.40.0       # Claude API + tool use
chromadb>=0.5.0         # 本地向量库
tavily-python>=0.3.0    # Web 搜索
rich>=13.0.0            # 终端美化
python-dotenv           # 环境变量管理
```

**环境变量：**
```
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
```

---

## 九、目录结构

```
my-agent/
├── main.py                  # 入口，Orchestrator 逻辑
├── agents/
│   ├── resume_agent.py
│   ├── research_agent.py
│   └── interview_agent.py
├── rag/
│   └── knowledge_base.py    # ChromaDB 初始化与检索
├── memory/
│   └── history.json         # 跨会话投递记录
├── knowledge/               # 用户个人经历文档
│   ├── resume_base.md
│   ├── experiences/
│   └── self_profile.md
├── outputs/                 # 每次运行的报告
├── .env
└── requirements.txt
```
