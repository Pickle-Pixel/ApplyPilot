# Stage 5 — User Role Discovery 覆盖报告

**生成日期**: 2026-11-07  
**输入**: Stage 4 user_journeys.json + 全量源码扫描  
**核心证据**: cli.py（所有命令无 user/role 参数）、config.py:200-223（get_tier 依赖检测）、database.py（无 users 表）

---

## 核心结论：ApplyPilot 是单角色系统

这个项目**不存在多用户角色体系**。经过全量源码扫描确认：

| 检查维度 | 结果 | 证据 |
|---------|------|------|
| 用户登录/注册 | ❌ 不存在 | grep `login\|register\|signup\|session` 全为 HTML/DOM 或 LLM API 字段 |
| 用户数据表 | ❌ 不存在 | SQLite 只有 jobs 表，无 users/roles/permissions |
| 角色切换逻辑 | ❌ 不存在 | 无 role switch / identity context |
| API 权限分级 | ❌ 不存在 | 所有命令参数与身份无关 |
| 页面/命令访问控制 | ⚠️ 存在但非角色驱动 | Tier 系统基于**依赖检测**，不是身份认证 |

---

## Tier 系统 ≠ 角色系统

| 维度 | Tier 系统 | 多角色系统 |
|------|----------|-----------|
| 决定因素 | 环境依赖是否安装 | 用户身份/权限 |
| 检测方式 | `shutil.which()` + 环境变量检测 | 登录 session + role lookup |
| 切换方式 | 装个依赖自动升级 | 重新登录 / 切换身份 |
| 同一用户能跨级？ | ✅ 是（同一人装更多东西解锁更多功能） | ❌ 通常不行 |
| 有单独的身份记录？ | ❌ 没有 | ✅ 有 users 表 |
| 代码证据 | config.py:200-223 | 不存在 |

**Playbook 对齐**：第 8 节明确说 "如果项目没有明确角色体系，不要强行创建角色"。ApplyPilot 只有一个人类角色 `job_seeker`，Tier 是能力维度而非身份维度。

---

## 角色 → 旅程 → CLI 完整映射

```
ROLE-0001: job_seeker (求职用户)
│
├── [Tier 1] 能用的命令（Python + pip 即可）
│   ├── init         → JOURNEY-0001
│   ├── doctor       → JOURNEY-0002
│   ├── run discover → JOURNEY-0003 STEP-0014, JOURNEY-0004
│   ├── run enrich   → JOURNEY-0003 STEP-0015, JOURNEY-0004
│   ├── status       → JOURNEY-0008 STEP-0032
│   ├── dashboard    → JOURNEY-0008 STEP-0033
│   └── apply --mark-applied/failed/reset → JOURNEY-0007
│
├── [Tier 2] 解锁（+ LLM API key）
│   ├── run score    → JOURNEY-0003 STEP-0016
│   ├── run tailor   → JOURNEY-0003 STEP-0017
│   ├── run cover    → JOURNEY-0003 STEP-0018
│   ├── run pdf      → JOURNEY-0003 STEP-0019
│   ├── run (all)    → JOURNEY-0003
│   └── run --stream → JOURNEY-0005
│
└── [Tier 3] 解锁（+ Claude Code CLI + Chrome + Node.js）
    ├── apply           → JOURNEY-0006
    ├── apply --gen     → JOURNEY-0009
    └── apply --url     → JOURNEY-0006 STEP-0025
```

**覆盖度**: 9 个旅程全部映射到 ROLE-0001。6 个 CLI 命令全部被覆盖。

---

## 系统内部执行组件（非人类角色）

源码中存在多个系统内部执行主体，但它们不是"用户角色"：

| ID | 组件 | 功能 | 证据 |
|----|------|------|------|
| COMP-0001 | Claude Code Agent | Tier 3 自动申请的 LLM 执行主体 | launcher.py:326-333 subprocess 调用 claude CLI |
| COMP-0002 | Playwright MCP | 浏览器操作桥接层 | launcher.py:60-72 MCP config |
| COMP-0003 | Chrome Worker | 独立 Chrome 实例 | chrome.py + launcher.py worker 循环 |
| COMP-0004 | Gmail MCP | 邮件验证码处理 | launcher.py:76-82 |
| COMP-0005 | JobSpy Engine | Tier 1 discover 第一来源 | discovery/jobspy_search.py |
| COMP-0006 | Workday Scraper | Tier 1 discover 第二来源 | discovery/workday.py |
| COMP-0007 | SmartExtract | Tier 1 discover 第三来源 | discovery/smartextract.py |

---

## Tier 检测逻辑源码级确认

```python
# config.py:200-223 — get_tier() 的完整检测链
def get_tier() -> int:
    # 第一步：检查 LLM API key 环境变量
    has_llm = any(os.environ.get(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_URL"))
    if not has_llm:
        return 1                    # Tier 1: 只有 discover/enrich/status

    # 第二步：检查 Claude CLI + Chrome
    has_claude = shutil.which("claude") is not None
    has_chrome = get_chrome_path()  # 抛 FileNotFoundError 则 = False

    if has_claude and has_chrome:
        return 3                    # Tier 3: 全功能 auto-apply

    return 2                        # Tier 2: AI 流水线但没有 auto-apply
```

**关键**：`shutil.which()` 是纯文件系统检查，`os.environ.get()` 是纯环境变量检查——**没有任何身份认证、session、user lookup**。

---

## 边界案例

| 场景 | 行为 | 代码位置 |
|------|------|---------|
| Tier 1 用户执行 `run all --dry-run` | dry-run 跳过 Tier gate，打印将执行的完整 stages 列表 | cli.py dry_run 分支在 pipeline.run_pipeline() 内部 |
| Tier 2 用户设置完 API key 后 | 同一个用户下次执行 `applypilot run` 自动进入 Tier 2 | config.py:209-223 无状态检测 |
| Tier 3 用户卸载 claude CLI | 下次执行 `applypilot apply` 会提示 Tier 不足 + 缺失项 | check_tier(3) 报错 |
| 两个不同机器的用户 | 各自独立，无任何跨机器协调机制 | 单用户单 SQLite DB 设计 |

---

## 下一步建议

1. **确认 Stage 5 结论** — 单角色系统（job_seeker），Tier 是能力维度而非角色
2. **进入 Stage 6** — CTO Dashboard / Final Deliverable。根据 Playbook，Stage 6 需要整合 Stage 1-5 的产物，生成最终的产品分析报告
3. **潜在优化点** — 如果未来要加多用户支持，需要引入 user_id、profile 归属、DB schema 变更等大量工作