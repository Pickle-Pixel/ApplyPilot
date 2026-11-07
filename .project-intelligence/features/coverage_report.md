# Stage 3 — Business Feature Discovery 覆盖报告

**生成日期**: 2026-11-07  
**输入**: Stage 2 atomic_capabilities.json (28 个能力)  
**输出**: Stage 3 business_features.json (10 个功能)

---

## 功能概览

| FEAT | 功能名 | 类型 | 可见性 | 包含能力数 | CLI 入口 |
|------|--------|------|--------|-----------|----------|
| FEAT-0001 | 首次安装与用户配置 | system_management | 用户可见 | 2 | `applypilot init` |
| FEAT-0002 | 系统健康诊断 | system_management | 用户可见 | 3 | `applypilot doctor` |
| FEAT-0003 | 多源职位发现 | user_facing | 用户可见 | 6 | `run --stage discovery` |
| FEAT-0004 | 职位详情丰富 | backend_automation | 后台自动化 | 6 | `run --stage enrichment` |
| FEAT-0005 | 职位匹配评分与筛选 | backend_automation | 后台自动化 | 2 | `run --stage scoring` |
| FEAT-0006 | 简历定制与 PDF 渲染 | backend_automation | 后台自动化 | 3 | `run --stage tailoring` |
| FEAT-0007 | 求职信生成与 PDF 渲染 | backend_automation | 后台自动化 | 3 | `run --stage cover_letter` |
| FEAT-0008 | 多 Worker 全自动职位申请 | user_facing | 用户可见 | 5 | `applypilot apply` |
| FEAT-0009 | 数据展示与实时监控 | user_facing | 用户可见 | 3 | `dashboard` / apply 期间 |
| FEAT-0010 | 一键流水线编排 | internal_technical | 用户可见 | 3 | `applypilot run` |

## 能力 → 功能追溯矩阵

| CAP | 能力名 | 归入功能 | 跨功能 |
|-----|--------|---------|--------|
| CAP-0001 | 配置加载与验证 | FEAT-0001 | ✅ 跨 8 个功能 |
| CAP-0002 | Chrome 二进制检测 | FEAT-0002 | — |
| CAP-0003 | SQLite 职位持久化 | FEAT-0003/04/05/08/09/10 | ✅ 跨 9 个功能 |
| CAP-0004 | 统一 LLM 调用 | FEAT-0002 (健康检查) | ✅ 跨 5 个业务功能 |
| CAP-0005 | 6 阶段流水线编排 | FEAT-0010 | — |
| CAP-0006 | CLI 命令分发 | FEAT-0010 | — |
| CAP-0007 | HTML 仪表板生成 | FEAT-0009 | — |
| CAP-0008 | 终端实时仪表板 | FEAT-0008/0009 | ✅ 跨 2 个功能 |
| CAP-0009 | JobSpy 多站爬虫 | FEAT-0003 | — |
| CAP-0010 | Workday ATS 爬虫 | FEAT-0003 | — |
| CAP-0011 | SmartExtract AI 提取 | FEAT-0003 | — |
| CAP-0012 | 代理字符串解析 | FEAT-0003 | — |
| CAP-0013 | 职位地点过滤 | FEAT-0003 | — |
| CAP-0014 | 详情页三层级联 | FEAT-0004 | — |
| CAP-0015 | ATS 平台检测 | FEAT-0004 | — |
| CAP-0016 | LLM 1-10 评分 | FEAT-0005 | — |
| CAP-0017 | 简历定制 | FEAT-0006 | — |
| CAP-0018 | 求职信生成 | FEAT-0007 | — |
| CAP-0019 | PDF 渲染 | FEAT-0006/0007 | ✅ 跨 2 个功能 |
| CAP-0020 | 内容验证规则引擎 | FEAT-0006/0007 | ✅ 跨 2 个功能 |
| CAP-0021 | 薪资字符串解析 | FEAT-0004 | — |
| CAP-0022 | 地点字符串解析 | FEAT-0004 | — |
| CAP-0023 | HTML 清洗与提取 | FEAT-0004 | — |
| CAP-0024 | Chrome 生命周期管理 | FEAT-0008 | — |
| CAP-0025 | 多 Worker 自动申请编排 | FEAT-0008 | — |
| CAP-0026 | Agent Prompt Builder | FEAT-0008 | — |
| CAP-0027 | 首次设置向导 | FEAT-0001 | — |
| CAP-0028 | 健康检查 | FEAT-0002 | — |

**覆盖率**: 28 / 28 能力已归类，unclassified: 0

---

## 跨功能复用网络

```
CAP-0003 (JobsDB) ──┬── FEAT-0003 发现
                     ├── FEAT-0004 丰富
                     ├── FEAT-0005 评分
                     ├── FEAT-0008 自动申请
                     ├── FEAT-0009 监控
                     └── FEAT-0010 流水线

CAP-0004 (LLM) ──────┬── FEAT-0003 (SmartExtract Phase 1)
                     ├── FEAT-0004 (Enrichment Tier 3)
                     ├── FEAT-0005 (Scoring)
                     ├── FEAT-0006 (Tailoring)
                     └── FEAT-0007 (Cover Letter)

CAP-0020 (Validation) ┬── FEAT-0006 (Resume validation)
                     └── FEAT-0007 (Cover letter validation)

CAP-0019 (PDF) ───────┬── FEAT-0006 (Resume PDF)
                     └── FEAT-0007 (Cover letter PDF)

CAP-0008 (Dashboard) ─┬── FEAT-0008 (Runtime real-time)
                     └── FEAT-0009 (Monitoring view)
```

---

## 关键发现

### 产品价值链条（用户视角）
```
┌─────────────────────────────────────────────────────┐
│  用户第一次启动：                                    │
│  FEAT-0001 init → 填配置 → FEAT-0002 doctor 诊断     │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  一键跑流水线：                                      │
│  FEAT-0010 run                                      │
│    ├── FEAT-0003 发现 (→ DB)                        │
│    ├── FEAT-0004 丰富 (→ DB)                        │
│    ├── FEAT-0005 评分 (→ DB)                        │
│    ├── FEAT-0006 定制 (→ PDF)                       │
│    └── FEAT-0007 求职信 (→ PDF)                     │
└────────────────────┬────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────┐
│  启动自动申请：                                      │
│  FEAT-0008 apply --workers 5 --score-min 7           │
│    ├── 实时 FEAT-0009 (Terminal Dashboard)           │
│    └── 事后 FEAT-0009 (HTML Dashboard)               │
└─────────────────────────────────────────────────────┘
```

### 流水线 vs 独立命令的关系
- `pipeline.py STAGES` 列表 = [discovery, enrichment, scoring, tailoring, cover_letter, apply]
- 但 `cli.py` 有独立的 `applypilot apply` 命令
- **推断**：`run` 命令可能包含 apply 阶段，或者 apply 设计为独立的"最后一公里" — 需 runtime 确认

### DB 双路径问题
- discovery 模块（FEAT-0003）使用 `sqlite3.connect` + `database.get_connection()` 直接写表
- JobsDB 类（CAP-0003）被 FEAT-0004/05/08/09/10 使用
- **所有写同一张 jobs 表** — 风险：连接池不一致、线程安全策略不统一、迁移逻辑只在 JobsDB 类中

### LLM 调用聚合成本
- CAP-0004 被 5 个业务功能复用
- discovery 阶段（FEAT-0003）只在 SmartExtract Phase 1 调用 LLM
- enrichment 阶段（FEAT-0004）在 Tier 3 兜底时调用
- scoring / tailoring / cover_letter 每个职位各一次 LLM 调用
- 一个职位完整流水线 ≈ 3-4 次 LLM 调用（+ SmartExtract 额外 1 次）

---

## 待人工确认问题

| # | 问题 | 影响 | 涉及功能 |
|---|------|------|---------|
| F3-Q1 | `run` 命令是否包含 apply 阶段？还是 apply 必须独立执行？ | 理解流水线完整范围 | FEAT-0010, FEAT-0008 |
| F3-Q2 | JobsDB 双路径写入 — 是否有意设计？ | 数据一致性、维护成本 | FEAT-0003 vs FEAT-0004/05/08 |
| F3-Q3 | CLI 是否支持阶段性跳过（如 `--skip-discovery`）？ | 用户灵活度 | FEAT-0010 |
| F3-Q4 | Wizard 保存前是否做 LLM API key 有效性验证？ | 用户体验（save 后才发现 key 无效） | FEAT-0001 |
| F3-Q5 | FEAT-0003 三个发现来源（JobSpy/Workday/SmartExtract）是否全部并行？还是顺序执行？ | 发现阶段总耗时 | FEAT-0003 |
| F3-Q6 | HTML Dashboard 是否支持自动刷新？还是需要手动重新 `dashboard` 命令？ | 监控便利性 | FEAT-0009 |
| F3-Q7 | 评分失败时的行为：fail-fast（标记为 failed）还是重试？ | 流水线健壮性 | FEAT-0005 |
| F3-Q8 | `--dangerously-skip-permissions` 安全边界如何定义？Agent 能访问本地文件到什么程度？ | 安全风险 | FEAT-0008 |

---

## 下一步建议

Stage 3 完成，10 个功能全部有明确的 CLI 入口、能力覆盖和代码证据。建议：

1. 确认 F3-Q1（run vs apply 关系）— 对理解 Stage 4 用户旅程至关重要
2. 确认 F3-Q5（三来源并行性）— 影响 FEAT-0003 的用户体验描述
3. 可直接进入 Stage 4（User Journey Reconstruction），此时需要结合 CLI 入口和真实的用户操作顺序来还原