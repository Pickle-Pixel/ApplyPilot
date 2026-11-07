# Stage 2 — Atomic Capability Index

**生成日期**: 2025-11-06  
**能力总数**: 28  
**状态分布**: 28 confirmed / 0 inferred / 0 unknown

---

## 层级视图

### 🟢 基础设施层 (CAP-0001 ~ CAP-0008)

| ID | 能力名 | 类型 | 输入 | 输出 | 依赖能力 |
|----|--------|------|------|------|----------|
| CAP-0001 | 用户配置加载 (YAML + .env + dataclass) | technical | ~/.applypilot/config.yaml + .env | AppConfig dataclass | — |
| CAP-0002 | Chrome 自动检测 + 版本读取 | technical | — | chrome_binary_path str | — |
| CAP-0003 | SQLite 持久化 (JobsDB) | technical | job dict | DB rows + 查询结果 | — |
| CAP-0004 | 统一 LLM 调用 (Gemini/OpenAI/Ollama) | technical | system_prompt, user_prompt, model | LLM 响应 str | — |
| CAP-0005 | 6 阶段流水线编排 (Pipeline class) | technical | stages list, mode (sequential/stream), resume, limit | 阶段执行结果 dict | CAP-0001, CAP-0003 |
| CAP-0006 | CLI 调度器 (6 commands) | technical | CLI args via typer | 各命令 side effect | CAP-0005, CAP-0027 |
| CAP-0007 | HTML 仪表板生成 | technical | DB job stats + scores | self-contained HTML file | CAP-0003 |
| CAP-0008 | 终端实时仪表板 | technical | worker dict (live updates) | Rich Live table + progress bar | — |

### 🟡 Tier 1 职位发现层 (CAP-0009 ~ CAP-0013)

| ID | 能力名 | 类型 | 输入 | 输出 | 依赖能力 |
|----|--------|------|------|------|----------|
| CAP-0009 | 多招聘网站爬虫 (JobSpy) | business | search_config list, sites list | DB jobs + per-query summary | CAP-0012, CAP-0013 |
| CAP-0010 | Workday ATS 企业门户爬虫 | business | employers.yaml dict | DB jobs + per-employer summary | CAP-0013 |
| CAP-0011 | AI 驱动通用网站职位提取 (SmartExtract) | business | target_urls or site templates | DB jobs | CAP-0004, CAP-0013 |
| CAP-0012 | 代理字符串解析 | utility | proxy_str: str | {host, port, user, pass, jobspy, playwright} | — |
| CAP-0013 | 职位地点过滤 | utility | location str, accept list, reject list | bool (keep/drop) | SearchConfig |

### 🔵 Tier 2 丰富 + 评分层 (CAP-0014 ~ CAP-0023)

| ID | 能力名 | 类型 | 输入 | 输出 | 依赖能力 |
|----|--------|------|------|------|----------|
| CAP-0014 | 职位详情页三层级联提取 | business | job URL list | enriched job dicts | CAP-0004 |
| CAP-0015 | ATS 平台识别 | utility | url: str | str ATS name or None | sites.yaml |
| CAP-0016 | 职位匹配评分 (LLM 1-10) | business | job dict + resume dict | ScoreResult (score, keywords, rationale) | CAP-0004 |
| CAP-0017 | 简历定制 (LLM JSON 输出) | business | job + base resume | TailoredResume JSON + saved file | CAP-0004 |
| CAP-0018 | 求职信生成 | business | job + resume + profile | formatted letter str | CAP-0004 |
| CAP-0019 | HTML/CSS 简历 + 求职信 PDF 渲染 | technical | resume dict + output path | PDF file | playwright |
| CAP-0020 | 内容验证规则引擎 | technical | tailored + base + profile + mode | ValidationReport (errors, warnings) | re stdlib |
| CAP-0021 | 薪资字符串解析 | utility | salary_text: str | {min, max, period, currency} | — |
| CAP-0022 | 地点字符串解析 | utility | loc_text: str | {city, state, country, remote} | — |
| CAP-0023 | HTML 清洗 + 文本/关键词提取 | utility | html: str | clean_html, plain_text, keywords | beautifulsoup4 |

### 🔴 Tier 3 自动申请层 (CAP-0024 ~ CAP-0028)

| ID | 能力名 | 类型 | 输入 | 输出 | 依赖能力 |
|----|--------|------|------|------|----------|
| CAP-0024 | Chrome 生命周期管理 (隔离 worker profile) | technical | worker_id, headless | CDP endpoint URL | CAP-0002 |
| CAP-0025 | 多 Worker 自动申请编排 (Claude Code) | business | jobs filter, workers count, dry_run | ApplySummary (success/fail counts) | CAP-0024, CAP-0003, CAP-0026, CAP-0008 |
| CAP-0026 | Apply Agent Prompt Builder (完整 SOP + CapSolver + Playwright) | technical | job, resume_path, cover_letter_path, chrome_cdp | full agent prompt str | — |
| CAP-0027 | 首次启动交互式设置向导 | technical | CLI args (optional) | AppConfig + written config files | CAP-0001 |
| CAP-0028 | 系统健康检查 (doctor 命令) | technical | fix: bool | Rich-formatted health table | CAP-0002, CAP-0004 |

---

## 依赖关系拓扑（高层）

```
CAP-0027 (Wizard)
  └─► CAP-0001 (Config)
        ├─► CAP-0005 (Pipeline)
        │     └─► CAP-0003 (JobsDB)
        ├─► CAP-0006 (CLI)
        │     ├─► CAP-0005 (Pipeline)
        │     │     └─► CAP-0009/010/011/014/016/017/018/025
        │     └─► CAP-0028 (Doctor)
        │           └─► CAP-0004 (LLM) + CAP-0002 (Chrome)
        ├─► CAP-0004 (LLM)
        │     ├─► CAP-0011 (SmartExtract)
        │     ├─► CAP-0014 (Enrichment T3)
        │     ├─► CAP-0016 (Scoring)
        │     ├─► CAP-0017 (Tailoring)
        │     └─► CAP-0018 (Cover Letter)
        ├─► CAP-0002 (Chrome Detection)
        │     └─► CAP-0024 (Chrome Lifecycle)
        │           └─► CAP-0025 (Apply Orchestration)
        │                 ├─► CAP-0026 (Apply Prompt)
        │                 ├─► CAP-0008 (Terminal Dashboard)
        │                 └─► CAP-0003 (JobsDB)
        └─► CAP-0007 (HTML Dashboard)
              └─► CAP-0003 (JobsDB)
```

## 能力类型分布

| 类型 | 数量 | 占比 |
|------|------|------|
| business（直接产品功能） | 9 | 32% |
| technical（技术支撑） | 12 | 43% |
| utility（通用工具） | 7 | 25% |

## 跨模块依赖发现（潜在重构机会）

1. **JobsDB 双访问路径**: discovery 模块（jobspy.py, workday.py, smartextract.py）直接 `sqlite3.connect()` + `database.get_connection()`，而 CLI/Pipeline/JobsDB 类封装了另一层。同一个 DB 表两套访问代码 → 考虑统一到 JobsDB 类。

2. **`_location_ok()` 重复实现**: jobspy.py 和 workday.py 各有一份相同逻辑的副本 → 提取到 config.py 或独立 utility 模块。

3. **`parse_proxy()` 定义在 jobspy.py** 却被 enrichment/detail.py 间接引用 → 应提取到 util 包。