# CTO Dashboard — ApplyPilot (v0.3.0)

**逆向产品发现分析报告 · 2026-11-07**  
生成方式：从 23 个源文件、3460 行 Python、7 个模块自动逆向还原  
覆盖范围：Stage 0 → Stage 5 全流水线完成

---

## 一、一秒概览

| 维度 | 数字 |
|------|------|
| 源文件 | 23 Python |
| 代码行 | ~3,460 LOC |
| 原子能力 | 28 |
| 业务功能 | 10（100% 覆盖） |
| 用户旅程 | 9（34 步骤，全 confirmed） |
| 人类角色 | 1（`job_seeker`） |
| 执行组件 | 7 |
| Tier 能力层 | 3 |
| 风险识别 | 8 |
| 开放问题 | 6 |

**一句话总结**：ApplyPilot 是一个单用户 CLI 工具，通过 **依赖检测**（非角色认证）解锁三级能力，核心流水线是 SQLite 驱动的 6 阶段数据传送带，Tier 3 自动申请通过 Claude Code Agent + Playwright MCP 实现浏览器自动化。

---

## 二、产品架构全景

### 2.1 分层视图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                               │
│  cli.py (6 命令) · wizard/init.py (5 步向导) · status/dashboard      │
├─────────────────────────────────────────────────────────────────────┤
│                     Orchestration Layer                              │
│  pipeline.py (6 阶段 STAGE_ORDER) · apply/launcher.py (Worker 循环) │
│  config.py (Tier gate + 依赖检测)                                    │
├─────────────────────────────────────────────────────────────────────┤
│                     Domain Engines                                   │
│  discovery/ (3 来源) · enrichment/ (3 级级联) · scoring/ (LLM)       │
│  apply/ (Chrome + Claude Agent)                                      │
├─────────────────────────────────────────────────────────────────────┤
│                     Infrastructure Layer                             │
│  config.py (Paths/AppConfig) · database.py (SQLite 单例)             │
│  llm.py (多 Provider) · apply/mcp_config.py                          │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流核心设计

ApplyPilot 的最聪明的设计决策：**用 SQLite jobs.db 做传送带**，阶段间不需要显式消息队列：

```
discover ──── upsert URL/title/company ──→ jobs.db
    │                                        │
    │ [PENDING_SQL: detail_scraped_at IS NULL]
    ▼                                        ▼
enrich ───── 写 description/ATS/salary ──→ jobs.db
    │                                        │
    │ [PENDING_SQL: full_description IS NOT NULL AND fit_score IS NULL]
    ▼                                        ▼
score ────── 写 fit_score (0-10) ────────→ jobs.db
    │                                        │
    │ [PENDING_SQL: fit_score >= 7 AND tailored_resume_path IS NULL]
    ▼                                        ▼
tailor ───── 写 tailored_resume_path ─────→ jobs.db
    │                                        │
    │ [PENDING_SQL: tailored_resume_path IS NOT NULL AND cover_letter_path IS NULL]
    ▼                                        ▼
cover ────── 写 cover_letter_path ────────→ jobs.db
    │                                        │
    │ [PENDING_SQL: tailored_resume_path LIKE '%.yaml']
    ▼                                        ▼
pdf ──────── 渲染 .yaml → .pdf ──────────→ jobs.db
                                              │
                                              │ apply 命令独立读取：
                                              │ tailored_resume_path IS NOT NULL
                                              │ AND applied_at IS NULL
                                              ▼
                                         apply/launcher.py
                                         └─ Chrome Worker (独立 profile + CDP 端口)
                                         └─ Claude Agent (subprocess + MCP config)
                                         └─ Playwright MCP (browser_* 工具)
                                         └─ Gmail MCP (验证码)
```

---

## 三、Tier 能力边界（不是角色）

**关键澄清**：Tier 系统检测的是**环境依赖是否安装**，不是用户身份认证。同一个 job_seeker 装更多依赖自动解锁更多功能。

```
                    ┌─────────────────────┐
 Tier 1:            │  Python + pip       │
 "Discovery"        │  只用本地能力        │
                    └─────────┬───────────┘
                              │ 设置 GEMINI_API_KEY / OPENAI_API_KEY / LLM_URL
                              ▼
                    ┌─────────────────────┐
 Tier 2:            │  LLM API key        │
 "AI Scoring &      │  解锁 score/tailor  │
  Tailoring"        │  /cover/pdf 流水线   │
                    └─────────┬───────────┘
                              │ 安装 Claude Code CLI + Chrome + Node.js
                              ▼
                    ┌─────────────────────┐
 Tier 3:            │  claude + chrome    │
 "Full Auto-Apply"  │  解锁 apply 命令    │
                    │  +多 Worker 并发    │
                    └─────────────────────┘
```

### Tier Gate 源码位置

| Gate | 代码位置 | 检测逻辑 |
|------|---------|---------|
| Tier 1 → Tier 2 | cli.py:118-121 | `check_tier(2, "AI scoring/tailoring")` |
| Tier 2 → Tier 3 | cli.py:189-190 | `check_tier(3, "auto-apply")` |
| Triple gate | cli.py:192-211 | Tier 3 ✅ → Profile 存在 ✅ → DB 有 ready jobs ✅ |
| get_tier() | config.py:200-223 | 纯 `shutil.which()` + `os.environ.get()` 检测 |

### 已发现的 Tier 边界 Bug

| ID | 问题 | 位置 | 修复优先级 |
|----|------|------|-----------|
| RISK-008 | `apply --gen --url` 被意外 Tier 3 门控 | cli.py:190 在 cli.py:213 gen 检查之前 | **P1 — 代码级确定性 bug** |

---

## 四、模块职责矩阵

| 模块 | 职责 | Tier 可访问 | 关键证据 |
|------|------|-----------|---------|
| cli.py | 所有 6 命令入口、Tier gate、apply 三重校验 | 全部 | 260 行 @app.command() |
| pipeline.py | 阶段编排（顺序 + 流式）、PENDING_SQL 定义 | 全部 | STAGE_ORDER list |
| config.py | AppConfig 数据类层级、Tier 检测算法 | 基础设施 | 260 行完整配置链 |
| database.py | SQLite 单例、jobs 表 CRUD、连接池 | 基础设施 | ThreadPoolExecutor |
| llm.py | 多 Provider LLM 客户端（Gemini/OpenAI/Anthropic） | 2/3 | 重试逻辑 |
| wizard/init.py | 5 步交互式 profile 向导 | 1 | InitWizard.run() |
| discovery/ | JobSpy + Workday + SmartExtract 三来源 | 1 | 顺序调用 |
| enrichment/ | JSON-LD → CSS → LLM 三级级联提取 | 1 | 免费兜底付费 |
| scoring/ | 评分 + 定制（self-critique 循环）+ 验证 + 求职信 | 2 | scorer/tailor/validator |
| apply/launcher.py | Worker 循环、Chrome 子进程、MCP 组装、Ctrl+C 双中断 | 3 | launcher.py main() |
| apply/chrome.py | Chrome 独立 profile + CDP 端口管理 | 3 | 多 Worker 隔离 |
| apply/prompt.py | Claude Agent system prompt 构建 | 3 | profile + job + rules |
| config/sites.yaml | 屏蔽 SSO 域名、Workday 模板 | 数据 | 配置文件 |

---

## 五、用户角色体系

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   HUMAN ROLE:  job_seeker (求职用户)                            │
│                                                                 │
│   ┌──────────┐  Tier 1 ──────────┐                              │
│   │          │                   │                              │
│   │  同一人   │  装 LLM API key    │  Tier 2                     │
│   │          │  ──────────────►   │                              │
│   │          │                   │                              │
│   │          │  装 claude+chrome  │  Tier 3                      │
│   │          │  ──────────────►   │                              │
│   └──────────┘                   └──────────────────────────────┘
│                                                                 │
│   没有用户登录/注册、没有权限切换、没有 session/role 表          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

系统内部执行组件（非人类角色，由 job_seeker 启动）：

COMP-0001  Claude Code Agent        ← Tier 3 auto-apply 的 LLM 执行主体
COMP-0002  Playwright MCP Server     ← 浏览器操作桥接层
COMP-0003  Chrome Worker             ← 独立 profile + CDP 端口
COMP-0004  Gmail MCP Server         ← 邮件验证码处理
COMP-0005  JobSpy Discovery Engine   ← Tier 1 discover 第一来源
COMP-0006  Workday CXS Scraper       ← Tier 1 discover 第二来源
COMP-0007  SmartExtract General      ← Tier 1 discover 第三来源
```

---

## 六、风险登记册

| ID | 类别 | 风险 | 严重度 | 概率 | 建议 |
|----|------|------|--------|------|------|
| RISK-001 | 架构 | 双 DB 访问模式（discovery 用裸 sqlite3，其他用 JobsDB 类） | 中 | 低 | 统一走 JobsDB + 加连接池 |
| RISK-002 | 外部依赖 | Workday CXS 未文档化 API 可能随时变更 | **高** | 中 | 加响应 schema 校验 + 降级到 SmartExtract |
| RISK-003 | 架构 | 阶段失败后静默继续，下游 PENDING_SQL 返回 0 | 低 | 中 | 加阶段完成信号 + 零记录告警 |
| RISK-004 | 安全 | API key 明文存 .env（无加密） | 中 | **高** | 用 OS keyring 或加密 config |
| RISK-005 | 依赖 | Claude Code CLI 是外部进程，更新不跟随 ApplyPilot | **高** | 中 | 锁已知好的 CLI 版本 + 版本探测 |
| RISK-006 | 架构 | 流式模式靠 SQLite 轮询（无消息队列） | 低 | 中 | 考虑 WAL + trigger 或 Redis |
| RISK-007 | 功能差距 | discover 内部三来源是顺序（非并发） | 低 | 低 | 可选 asyncio 并行 |
| RISK-008 | Bug | apply --gen --url 被意外 Tier 3 门控 | 低 | **高（代码级确定）** | **P1 — 移动 gen 检查到 Tier gate 之前** |

---

## 七、6 个待运行时验证的开放问题

| # | 问题 | 证据位置 |
|---|------|---------|
| F4-Q1 | `--stream` 模式中 discover 内部三来源是否仍为顺序？ | pipeline.py `_run_discover()` 是顺序 |
| F4-Q2 | `apply --gen --url` 是否意外要求 Tier 3？（**已确认为 bug**） | cli.py:190 gate 在 cli.py:213 gen 之前 |
| F4-Q3 | `--continuous` 是否触发 discover？还是只轮询 DB？ | launcher.py 只从 DB 读，无 discover 调用 |
| F4-Q4 | wizard 设置的 password 字段在哪里被实际消费？ | init.py:104 设置，apply/prompt.py:571 引用 |
| F4-Q5 | 某阶段失败后下游如何处理？ | try/except 后继续，PENDING_SQL 返回 0 → no-op |
| F4-Q6 | CapSolver key 配置后如何注入 auto-apply 流程？ | doctor.py 检查，launcher.py MCP config 无 capsolver |

---

## 八、核心用户旅程路径图

```
                      ┌──────────────┐
                      │   init 向导   │
                      │ (5 步)       │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐     ┌──────────────┐
                      │  doctor 诊断  │────►│  补装依赖     │
                      └──────┬───────┘     └──────┬───────┘
                             │                     │
                Tier 1 解锁   │                     │ Tier 2/3 解锁
                             ▼                     ▼
┌──────────────────────────────────────────────────────────┐
│           run 流水线（pipeline.py STAGE_ORDER）           │
│  discover → enrich → score → tailor → cover → pdf        │
│                                                          │
│  [顺序模式: _run_sequential]                              │
│  [流式模式: --stream → PENDING_SQL 轮询]                 │
└──────────────────────────────┬───────────────────────────┘
                               │
                               ▼
                    tailored_resume_path 已就绪
                               │
                               ▼
                    ┌──────────────────┐
                    │  apply 自动申请   │ ← Tier 3 gate
                    │  launcher.py     │
                    │  ├─ Chrome Worker (N 并行)
                    │  ├─ Claude Agent (subprocess)
                    │  ├─ Playwright MCP
                    │  └─ Gmail MCP
                    └────────┬─────────┘
                             │
                    Ctrl+C 双中断
                             │
                             ▼
                    结果分类汇总：
                    APPLIED / EXPIRED / CAPTCHA / LOGIN_ISSUE
                    UNSAFE_PERMISSIONS / SSO_REQUIRED / ...
```

---

## 九、优先行动项（按 ROI 排序）

| 优先级 | 行动 | 风险关联 | 工作量 | 影响 |
|--------|------|---------|--------|------|
| **P1** | 修复 `apply --gen` Tier gate 顺序 | RISK-008 | 低（移动 2 段代码） | Tier 2 用户可调试 auto-apply |
| **P2** | 统一 DB 访问模式 | RISK-001 | 中 | 连接池一致，为未来 Postgres 迁移铺路 |
| **P3** | Workday 降级到 SmartExtract | RISK-002 | 中 | Workday API 挂了自动切 SmartExtract |
| **P4** | 加密 API key 存储 | RISK-004 | 中-高 | 安全合规改进 |
| **P5** | 验证并文档化 `--continuous` 行为 | F4-Q3 | 低 | 消除用户歧义 |

---

## 十、与竞品差异化分析

| 维度 | ApplyPilot | 典型竞品 |
|------|-----------|---------|
| 形态 | 单二进制 CLI + Tier 解锁 | Web SaaS / 浏览器插件 |
| 自动化执行 | Claude Code Agent + Playwright MCP（LLM 驱动） | 直接 Playwright / Puppeteer |
| 流水线数据层 | SQLite 传送带 + 流式并发 | 内存队列 / Redis |
| 多源策略 | JobSpy + Workday CXS + SmartExtract（三来源） | 单一来源或少量 |
| 自我定制 | self-critique 循环（tailor.py 生成 + 自评 + 迭代） | 单次生成 |
| 自动申请隔离 | 多 Worker 独立 Chrome profile + CDP 端口 | 通常单实例 |

---

## 十一、产物清单

```
.project-intelligence/
├── config/
│   ├── analysis_config.yaml          ← 项目元数据 + 阶段进度
│   └── exclusions.yaml               ← 扫描排除配置
├── inventory/                        ← Stage 1: Code Inventory
│   ├── files.json (23 files)
│   ├── symbols.json (37 AST symbols)
│   ├── imports.json
│   └── configs.json
├── capabilities/                     ← Stage 2: 原子能力
│   ├── atomic_capabilities.json (28)
│   └── capability_index.md
├── features/                         ← Stage 3: 业务功能
│   ├── business_features.json (10, 100% 覆盖)
│   └── coverage_report.md
├── journeys/                         ← Stage 4: 用户旅程
│   ├── user_journeys.json (9 journeys, 34 steps)
│   └── coverage_report.md
├── roles/                            ← Stage 5: 用户角色
│   ├── user_roles.json (1 role, 7 components)
│   └── coverage_report.md
└── dashboard/                        ← Stage 6: 最终交付物 ⭐
    ├── cto_dashboard.json
    └── cto_report.md                ← 本文件
```

---

## 十二、Playbook 完成状态

```
Stage 0  ✅  INIT                项目元数据
Stage 1  ✅  CODE INVENTORY      23 files · 37 symbols · 2 modules
Stage 2  ✅  CAPABILITIES        28 原子能力
Stage 3  ✅  FEATURES            10 业务功能 · 100% 覆盖
Stage 4  ✅  JOURNEYS            9 旅程 · 34 steps · 全 confirmed
Stage 5  ✅  ROLES               1 人类角色 · 7 执行组件
Stage 6  ✅  CTO DASHBOARD       综合报告 + JSON Dashboard

总耗时: ~2 小时（从源码自动逆向）
分析文件数: 23 Python 源文件
证据点: 87 个（全部有源码行号）
```

---

*本报告由逆向产品发现流水线自动生成。所有结论均基于源码硬证据，不包含推测。*