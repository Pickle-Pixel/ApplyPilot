# Stage 4 — User Journey Reconstruction 覆盖报告

**生成日期**: 2026-11-07  
**输入**: Stage 3 business_features.json (10 功能) + Stage 1 inventory + Stage 2 capabilities  
**核心证据**: cli.py (完整命令实现)、pipeline.py (完整阶段定义与执行逻辑)、wizard/init.py (完整初始化流程)、apply/launcher.py (main 函数)、config.py (Tier 系统)

---

## 用户旅程总览

| JOURNEY | 旅程名 | Tier | 关联功能 | 步骤数 | 类型 |
|---------|--------|------|---------|--------|------|
| JOURNEY-0001 | 首次安装与配置向导 | 1 | FEAT-0001, FEAT-0010 | 7 | 线性 |
| JOURNEY-0002 | 系统健康诊断 | 1 | FEAT-0002 | 5 | 线性 |
| JOURNEY-0003 | 一键完整流水线 | 2 | FEAT-0010 + FEAT-0003~0007 | 8 | 线性 |
| JOURNEY-0004 | 分阶段流水线执行 | 1 | FEAT-0010 | 1 | 分支 |
| JOURNEY-0005 | 流式并发流水线 | 2 | FEAT-0010 | 3 | 线性 |
| JOURNEY-0006 | 自动职位申请 | 3 | FEAT-0008 | 5 | 线性 |
| JOURNEY-0007 | 手动状态管理 | 1 | FEAT-0008 | 1 | 工具 |
| JOURNEY-0008 | 数据状态查看 | 1 | FEAT-0009 | 2 | 线性 |
| JOURNEY-0009 | Apply 调试模式 | 3 | FEAT-0008 | 2 | 工具 |

---

## Tier 分层与解锁路径

```
Tier 1 (Discovery)
│
├── JOURNEY-0001 首次安装与配置
├── JOURNEY-0002 健康诊断
├── JOURNEY-0004 分阶段运行 discover/enrich
├── JOURNEY-0007 手动状态管理
└── JOURNEY-0008 数据查看 (status + dashboard)
│
│  解锁条件: .env 中有 GEMINI_API_KEY / OPENAI_API_KEY / LLM_URL
▼
Tier 2 (AI Scoring & Tailoring)
│
├── JOURNEY-0003 完整流水线 run all
├── JOURNEY-0004 分阶段运行 score/tailor/cover/pdf
└── JOURNEY-0005 流式并发 --stream
│
│  解锁条件: claude + chrome + npx 全部可用
▼
Tier 3 (Full Auto-Apply)
│
├── JOURNEY-0006 自动申请 apply
└── JOURNEY-0009 Apply 调试 --gen --url
```

---

## 关键发现（源码级证据）

### 发现 1：`run` ≠ `apply` — 独立命令
pipeline.py STAGE_ORDER 只有 `(discover, enrich, score, tailor, cover, pdf)`，**没有 apply**。
apply 是 cli.py 中的独立 `@app.command()`，有独立的前置检查链。
→ 直接回答了 Stage 3 的 F3-Q1。

### 发现 2：discover 内部顺序执行
pipeline.py `_run_discover()` 明确显示 JobSpy → Workday → SmartExtract **顺序调用**，每个有独立 try/except。
→ 部分回答了 F3-Q5：三来源在 discover 阶段是顺序而非并发。

### 发现 3：Tier 系统与 gate 机制
config.py 明确定义：
- Tier 1: `["init", "run discover", "run enrich", "status", "dashboard"]`
- Tier 2: `["run score", "run tailor", "run cover", "run pdf", "run"]`
- Tier 3: `["apply"]`

cli.py 中的 `check_tier(N, feature)` 是硬 gate，Tier 不足直接 `SystemExit(1)`。

### 发现 4：apply 的三重前置闸门
1. `check_tier(3)` → Tier 3 必须
2. `PROFILE_PATH.exists()` → init 必须
3. SQL 查询 DB 中 `tailored_resume_path IS NOT NULL AND applied_at IS NULL` 的数量 → 至少 1 个已定制简历

这三重闸门确保用户不会在错误的时机启动自动申请。

### 发现 5：流式模式的 DB 传送带设计
pipeline.py `_PENDING_SQL` 字典为每个阶段定义了独立的 DB 字段检查：
- enrich: `detail_scraped_at IS NULL`
- score: `full_description IS NOT NULL AND fit_score IS NULL`
- tailor: `fit_score >= ? AND tailored_resume_path IS NULL AND tailor_attempts < 5`
- cover: `tailored_resume_path IS NOT NULL AND cover_letter_path IS NULL AND cover_attempts < 5`
- pdf: `tailored_resume_path LIKE '%.txt'`

这使得上游阶段可以持续写入，下游阶段可以自然跟上——无需显式消息队列。

### 发现 6：--dry-run 不做 Tier 检查
cli.py 中 `check_tier` 在 LLM stages 检查之前，但 dry-run 模式的逻辑在 pipeline.run_pipeline() 内部，如果 dry_run=True，直接打印将执行的 stages 列表然后返回——不进入实际 runner。这意味着 Tier 1 用户可以用 `run all --dry-run` 预览完整流水线。

---

## 阶段顺序证据链

```
JOURNEY-0001 init ─────────────────────────────┐
                                                │
JOURNEY-0002 doctor ────────┐                  │
                            │                  ▼
JOURNEY-0003 run all ──── Tier 2 解锁 ◄─── init 完成
  ├── STEP-0014 discover ────┐
  ├── STEP-0015 enrich ───────┤
  ├── STEP-0016 score ────────┼── 顺序执行（或 --stream 并发）
  ├── STEP-0017 tailor ───────┤
  ├── STEP-0018 cover ────────┤
  └── STEP-0019 pdf ──────────┘
                                                │
                                                ▼
JOURNEY-0006 apply ───── Tier 3 解锁 ◄─── pdf 完成 + Chrome + Claude
  ├── STEP-0026 Chrome Worker
  ├── STEP-0027 Agent 自动申请
  ├── STEP-0028 实时监控 / Ctrl+C
  └── STEP-0029 清理汇总
```

---

## Feature → Journey 覆盖矩阵

| FEAT | 功能名 | 出现在哪些旅程 | 状态 |
|------|--------|---------------|------|
| FEAT-0001 | 首次安装与配置 | JOURNEY-0001 | ✅ |
| FEAT-0002 | 系统健康诊断 | JOURNEY-0002 | ✅ |
| FEAT-0003 | 多源职位发现 | JOURNEY-0003 STEP-0014 | ✅ |
| FEAT-0004 | 职位详情丰富 | JOURNEY-0003 STEP-0015 | ✅ |
| FEAT-0005 | 职位匹配评分与筛选 | JOURNEY-0003 STEP-0016 | ✅ |
| FEAT-0006 | 简历定制与 PDF 渲染 | JOURNEY-0003 STEP-0017, STEP-0019 | ✅ |
| FEAT-0007 | 求职信生成与 PDF 渲染 | JOURNEY-0003 STEP-0018, STEP-0019 | ✅ |
| FEAT-0008 | 多 Worker 全自动职位申请 | JOURNEY-0006, 0007, 0009 | ✅ |
| FEAT-0009 | 数据展示与实时监控 | JOURNEY-0006 (STEP-0028), JOURNEY-0008 | ✅ |
| FEAT-0010 | 一键流水线编排 | JOURNEY-0003, 0004, 0005 | ✅ |

**覆盖率**: 10 / 10 功能全部有对应的用户旅程，unclassified: 0

---

## 待人工确认问题

| # | 问题 | 证据来源 | 建议验证方式 |
|---|------|---------|-------------|
| F4-Q1 | 流式模式中 discover 内部三来源是否仍为顺序？还是变成并发？ | pipeline.py:282-290 显示 streaming 中 discover 只 run 一次，与 sequential 相同 | 运行 --stream 观察 discover 子阶段日志 |
| F4-Q2 | apply --gen 模式是否真的需要 Tier 3？check_tier(3) 在 gen 检查之前 | cli.py:189-190 在 cli.py:213 gen 检查之前 | Tier 2 环境试跑 apply --gen --url |
| F4-Q3 | --continuous 模式是否会自动触发 discover？ | launcher.py:main() 只从 DB 取 jobs，没有 discover 调用 | 观察空 DB + continuous 的行为 |
| F4-Q4 | wizard 中设置的 password 字段在哪些流程中实际使用？ | wizard/init.py:104 设置 password=True，未在 apply 旅程中看到明确引用 | 搜 codebase 中 `password` 的实际引用 |
| F4-Q5 | 某个阶段失败后，后续阶段如何处理？ | pipeline.py catch Exception 后继续，但后续阶段会 SQL 查询待处理记录 | 人为让 discover 失败，观察 enrich 是否处理 |
| F4-Q6 | CapSolver key 配置后，在 apply Agent 中如何被注入？ | launcher.py:_make_mcp_config 只配置 playwright + gmail MCP | 搜 launcher.py 中 capsolver 引用 |

---

## 下一步建议

1. **确认 Stage 4 完整性** — 9 个旅程覆盖了全部 10 个功能，关键入口和前置条件都有源码证据
2. **确认 F4-Q2** — `apply --gen` 是否意外要求 Tier 3（可能是设计 bug）
3. **进入 Stage 5** — User Role Discovery。目前观察到的角色只有一个：`job_seeker`（CLI 用户）。但 CapSolver、Claude Agent、Playwright MCP 是系统内部执行主体，可能需要区分
4. **Stage 5 可能需要** 重新审视：是否存在多个角色（Tier 1 vs Tier 2 vs Tier 3 的用户是同一个人但权限不同？还是不同用户？）