# ApplyPilot 逆向产品发现：AI 执行手册（初版）

版本：0.1\
目标：让一个 Coding Agent 在不要求用户先理解整个代码库的情况下，从现有源码中逐步还原产品功能结构，并最终生成以用户旅程为主索引的 CTO Dashboard。

------------------------------------------------------------------------

## 0. 项目目标与边界

### 0.1 最终目标

建立一套可重复执行的 Reverse Product Discovery（逆向产品发现）流程：

``` text
现有源码
  ↓
代码资产清单（Code Inventory）
  ↓
最小功能能力（Atomic Capabilities）
  ↓
业务功能（Business Features）
  ↓
用户旅程（User Journeys）
  ↓
用户角色（User Roles）
  ↓
CTO Dashboard
```

分析顺序必须从底层向上。最终 Dashboard 的浏览顺序则从用户角色向下：

``` text
用户角色 → 用户旅程 → 用户操作/功能 → 代码入口 → 相关代码
```

### 0.2 核心使用场景

用户想修改某个产品功能时，可以：

1.  选择用户角色；
2.  沿着用户实际使用流程找到目标操作；
3.  查看该操作对应的代码入口；
4.  查看入口涉及的主要文件、类、函数和配置；
5.  决定从哪里开始阅读或修改。

### 0.3 非目标

本项目不是以代码目录为主视图的依赖图工具，也不要求第一轮就完整解释所有调用链。函数调用关系、依赖关系和数据流属于辅助信息。

### 0.4 基本原则

-   先盘点事实，再推断业务含义。
-   分阶段执行，不要求模型一次性理解整个仓库。
-   每阶段结果必须持久化到本地。
-   每个结论都要能追溯到文件、代码符号、文档或测试证据。
-   明确区分 `confirmed`、`inferred`、`unknown`、`needs_review`。
-   不得把文件名或目录名直接当成业务事实。
-   不得虚构用户角色、页面、API、功能或代码入口。
-   允许逐步修正；修正必须保留来源和变更记录。
-   分析过程与 Dashboard 展示分离：先建立知识库，再生成界面。

------------------------------------------------------------------------

## 1. 建议交付物与目录

所有分析产物放在仓库根目录下的独立目录中，不修改原始业务代码。

``` text
.project-intelligence/
├── README.md
├── config/
│   ├── analysis_config.yaml
│   └── exclusions.yaml
├── inventory/
│   ├── files.json
│   ├── symbols.json
│   ├── imports.json
│   └── configs.json
├── capabilities/
│   └── atomic_capabilities.json
├── features/
│   └── business_features.json
├── journeys/
│   └── user_journeys.json
├── roles/
│   └── user_roles.json
├── evidence/
│   ├── evidence_index.json
│   └── unresolved_questions.json
├── reports/
│   ├── coverage_report.md
│   └── analysis_log.md
└── dashboard/
    └── （后续阶段生成）
```

如果实际项目适合 SQLite，可以在后续阶段增加数据库；初期 JSON 文件便于检查、版本控制和人工修订。不要在第一阶段为了技术选型而过度设计。

------------------------------------------------------------------------

## 2. 全局执行规则（Master Prompt）

将以下内容作为 Coding Agent 的总提示词：

> 你是一个谨慎的代码库逆向分析代理。你的任务不是直接总结整个项目，也不是根据目录名称猜测产品功能，而是按照本手册定义的阶段，从现有代码和项目证据中逐步重建产品结构。
>
> 严格按照以下阶段执行：
>
> 1.  Code Inventory：盘点文件、代码符号、参数、返回类型、导入关系和配置项。
> 2.  Atomic Capability Extraction：基于实际实现，识别代码提供的最小能力。
> 3.  Business Feature Discovery：把相关能力归纳为业务功能。
> 4.  User Journey Reconstruction：结合 UI、CLI、API、文档和测试，还原用户操作及其流程。
> 5.  User Role Discovery：根据可验证的访问入口和用户流程识别角色。
> 6.  Dashboard Preparation：整理适合用户角色 → 用户旅程 → 用户操作 → 代码入口的展示数据。
>
> 不得跳过阶段，不得在前一阶段未完成或未达到质量门槛时擅自进入下一阶段。每阶段开始前先读取已有产物，判断是否可以复用；每阶段结束后写入文件并验证 JSON 格式。
>
> 对大仓库按目录或文件批次处理。每批记录处理范围、成功数量、失败数量和待处理项目。不得假设一次上下文能够容纳整个仓库。
>
> 所有事实必须带证据。代码证据尽可能精确到文件路径、符号名和行号范围；文档证据记录文档路径和标题；测试证据记录测试文件和测试名称。
>
> 对结论使用以下状态： 
> - `confirmed`：有直接证据支持；
> - `inferred`：根据多个证据推断，但没有直接声明；
> - `unknown`：现有材料无法判断；
> - `needs_review`：存在冲突或需要人工确认。
>
> 置信度不能代替证据。没有证据时不得仅凭高置信度把结论标记为 confirmed。
>
> 不得虚构功能、角色、用户界面、API、文件路径或代码入口。推断必须明确标记，并说明依据。
>
> 第一阶段以结构提取为主，不要提前生成用户旅程。后续阶段也不得把技术内部步骤误写成用户操作。
>
> 未经用户明确要求，不要修改、重构或格式化原始源码；不要执行有副作用的命令；不要访问或输出密钥、令牌、密码及 `.env` 中的敏感值。
>
> 每次执行只完成用户指定的阶段或批次。完成后报告结果、产物路径、覆盖情况、未解决问题和下一步建议，等待用户确认后再继续。

------------------------------------------------------------------------

## 3. Stage 0：项目识别与分析准备

### 目标

确认分析对象、技术栈、仓库范围和排除规则，避免误扫虚拟环境、构建产物或第三方依赖。

### Prompt

> 请执行 Stage
> 0：项目识别与分析准备。现在不要分析业务功能，也不要生成用户旅程。
>
> 1.  确认仓库根目录和 Git 状态。
> 2.  识别主要语言、包管理文件、应用入口、测试目录、文档目录和配置文件。
> 3.  识别应排除的目录，例如虚拟环境、缓存、构建输出、依赖目录、Git 元数据和大型生成文件。排除前列出规则和命中路径。
> 4.  不要读取或输出密钥、令牌、密码及 `.env` 中的敏感值；只记录配置文件存在及安全的键名（如确有必要）。
> 5.  创建 `.project-intelligence/` 目录及初始配置。
> 6.  输出项目概览、纳入/排除范围、待确认事项。
>
> 如果仓库路径不明确、存在多个项目根目录，或排除规则可能漏掉业务源码，请先提问，不要猜测。

### 验收条件

-   仓库根目录已确认；
-   扫描范围和排除规则可审查；
-   没有泄露敏感值；
-   目录和配置文件已创建；
-   尚未进行业务功能推断。

------------------------------------------------------------------------

## 4. Stage 1：Code Inventory（代码资产清点）

### 目标

建立完整、可追溯的代码目录与符号清单。此阶段以静态结构提取为主，不要求
AI 深入解释业务逻辑。

### 需要提取的信息

**文件级：**

-   相对路径；
-   文件类型、语言；
-   文件大小（可选）；
-   是否为源码、测试、文档、配置或脚本；
-   是否成功解析；
-   解析器或提取方式；
-   文件哈希或更新时间（供增量分析使用，可选）。

**符号级：**

-   模块、类、函数、方法；
-   所属文件和父级符号；
-   参数名称、默认值、类型注解；
-   返回类型注解；
-   Docstring 或简短原始说明；
-   可获取时的起止行号；
-   可见性或装饰器（如有）。

**关系级：**

-   import / require / include 等静态引用；
-   配置文件与配置键；
-   测试文件与被测符号的候选关联（只记录证据，不强行断言）。

### Prompt

> 请执行 Stage 1：Code Inventory。只提取代码结构，不归纳业务功能，不推断用户角色或用户旅程。
>
> 先读取 Stage 0 的扫描范围和排除规则。然后按目录分批扫描；每批完成后立即保存结果，记录批次状态，避免依赖长上下文。
>
> 对支持 AST 的语言优先使用确定性解析器提取符号；不要仅靠 LLM 从源码文本中猜函数和参数。对无法可靠解析的语言，标记解析方式和局限。
>
> 为每个文件记录路径、类型、分类和解析状态。为每个类、函数和方法记录名称、所属文件、父级、参数、默认值、类型注解、返回类型注解、Docstring 和行号（可获取时）。
>
> 记录静态导入关系，但不要把 import 关系等同于运行时调用关系。
>
> 配置文件只提取安全的配置键、类型或结构，不读取或输出密钥和敏感值。
>
> 完成每个批次后更新 JSON，并验证格式。全部批次结束后生成覆盖报告，列出未扫描、解析失败、跳过及原因。
>
> 不得在本阶段生成 Atomic Capabilities、Business Features、User Journeys 或 User Roles。

### 建议输出结构

`files.json`

``` json
{
  "schema_version": "1.0",
  "files": [
    {
      "file_id": "FILE-0001",
      "path": "src/example.py",
      "language": "python",
      "category": "source",
      "parse_status": "parsed",
      "symbols": ["SYM-0001"],
      "evidence": {
        "method": "python_ast",
        "line_start": 1,
        "line_end": 120
      }
    }
  ]
}
```

`symbols.json`

``` json
{
  "schema_version": "1.0",
  "symbols": [
    {
      "symbol_id": "SYM-0001",
      "kind": "function",
      "name": "parse_document",
      "file_id": "FILE-0001",
      "qualified_name": "parse_document",
      "parameters": [
        {
          "name": "content",
          "type_annotation": "str",
          "default": null
        }
      ],
      "return_type": "dict",
      "docstring": null,
      "line_start": 10,
      "line_end": 28
    }
  ]
}
```

以上仅为 Schema 示例，实际类型和字段应根据语言适配。

### 验收条件

-   扫描范围内的文件都有记录，或有明确跳过原因；
-   支持语言的符号提取使用可靠解析方式；
-   参数和返回类型没有注解时记录为 `unknown`，不猜测；
-   所有符号都能反查到文件；
-   JSON 可解析，ID 唯一；
-   生成覆盖率和失败清单。

------------------------------------------------------------------------

## 5. Stage 2：Atomic Capability Extraction（最小功能能力提取）

### 目标

从代码元素中识别"代码实际提供的能力"。能力不是简单的函数名，也不一定与单个函数一一对应。

### Prompt

> 请执行 Stage 2：Atomic Capability Extraction。输入仅使用 Stage 1 的清单，并在需要判断行为时读取对应的有限源码片段。
>
> 逐批分析函数、方法、类及相关配置。对每个候选能力，说明它做什么、输入什么、输出什么、产生什么副作用（如文件写入、数据库操作、网络请求），以及它依赖的其他代码元素。
>
> 不要仅凭函数名或文件名判断能力。至少检查实现、Docstring、调用位置、测试或文档中的一种直接证据；重要结论尽量交叉验证。
>
> 区分业务能力、技术基础设施能力和通用工具能力。不要把日志、格式转换、通用字符串处理等工具函数强行包装成用户功能。
>
> 一个能力可以由多个代码元素共同实现；一个代码元素也可能参与多个能力。保留多对多关系。
>
> 每个能力必须包含证据、状态和未解决问题。证据不足时标记 `unknown` 或
> `needs_review`，不要编造。
>
> 按批次保存 `atomic_capabilities.json`，并生成待验证清单。

### 建议输出结构

``` json
{
  "capability_id": "CAP-0001",
  "name": "Document Text Extraction",
  "description": "Extracts text content from an uploaded document.",
  "capability_type": "technical",
  "source_symbols": ["SYM-0001"],
  "inputs": [],
  "outputs": [],
  "side_effects": [],
  "dependencies": [],
  "status": "inferred",
  "evidence": [],
  "open_questions": []
}
```

------------------------------------------------------------------------

## 6. Stage 3：Business Feature Discovery（业务功能归纳）

### 目标

把相关的最小能力组织成产品层面的业务功能，同时保留能力与代码之间的追溯关系。

### Prompt

> 请执行 Stage 3：Business Feature Discovery。读取已完成的 Atomic Capabilities，并结合必要的 UI、CLI、API、文档和测试证据。
>
> 将相关能力归纳为业务功能。每个业务功能必须说明用户或系统要达成的目标、包含哪些能力、涉及哪些代码元素，以及判断依据。
>
> 不要为了形成整齐的模块而强行分组。无法归类的能力进入 `unclassified`，不要丢弃。
>
> 区分用户可见功能、后台自动化、系统管理能力和内部技术能力。后台任务不自动等于用户操作。
>
> 对功能之间的依赖只记录有证据支持的关系。保留一个能力被多个业务功能复用的可能性。
>
> 输出 `business_features.json`、功能覆盖报告和待人工确认问题。

### 建议输出结构

``` json
{
  "feature_id": "FEAT-0001",
  "name": "Resume Processing",
  "description": "Processes resume content into structured information.",
  "feature_type": "user_facing",
  "capability_ids": ["CAP-0001", "CAP-0002"],
  "entry_candidates": [],
  "related_features": [],
  "status": "inferred",
  "evidence": [],
  "open_questions": []
}
```

------------------------------------------------------------------------

## 7. Stage 4：User Journey Reconstruction（用户旅程还原）

### 目标

从已识别的业务功能和真实交互入口中，还原用户可感知的操作步骤及其顺序。

### Prompt

> 请执行 Stage 4：User Journey Reconstruction。不要把内部函数调用链直接当作用户旅程。
>
> 结合页面、组件、路由、API、CLI 命令、文档、测试和业务功能，识别用户能够执行或感知的操作。
>
> 尽量还原操作的前置条件、触发方式、成功结果、失败分支和下一步操作。区分用户主动操作、系统自动行为和后台维护行为。
>
> 对流程顺序，优先使用 UI 导航、路由、状态转换、测试场景和文档证据。仅凭代码文件排列或函数调用顺序不能证明用户旅程顺序。
>
> 证据不足以确定完整流程时，输出候选流程并标记 `inferred` 或 `needs_review`。不要把推测呈现为事实。
>
> 一个业务功能可以出现在多个旅程中；同一操作也可能被多个角色使用。
>
> 输出 `user_journeys.json`，并列出缺少 UI 或产品证据的步骤。

### 建议输出结构

``` json
{
  "journey_id": "JOURNEY-0001",
  "name": "Prepare a Resume",
  "actor_candidates": [],
  "steps": [
    {
      "step_id": "STEP-0001",
      "order": 1,
      "user_action": "Upload Resume",
      "feature_ids": ["FEAT-0001"],
      "trigger_evidence": [],
      "outcome": null,
      "status": "needs_review"
    }
  ],
  "status": "inferred",
  "evidence": [],
  "open_questions": []
}
```

------------------------------------------------------------------------

## 8. Stage 5：User Role Discovery（用户角色识别）

### 目标

根据真实访问入口、权限边界和用户旅程识别角色，而不是凭产品常识假定一定存在管理员或运营人员。

### Prompt

> 请执行 Stage 5：User Role Discovery。综合认证逻辑、权限检查、页面访问控制、API 授权、文档和用户旅程，识别项目中有证据支持的用户角色。
>
> 对每个角色说明其证据、可访问的旅程和操作。区分真实角色、权限等级、匿名访问者、系统服务账号和后台执行主体。
>
> 如果项目没有明确角色体系，不要强行创建角色；可以使用 `role_unknown` 或记录候选角色并标记待验证。
>
> 允许一个角色拥有多个旅程，一个旅程也可能由多个角色参与。
>
> 输出 `user_roles.json`，并列出角色定义冲突或证据不足之处。

### 建议输出结构

``` json
{
  "role_id": "ROLE-0001",
  "name": "Candidate Role",
  "description": "A user category supported by project evidence.",
  "journey_ids": ["JOURNEY-0001"],
  "permissions": [],
  "status": "inferred",
  "evidence": [],
  "open_questions": []
}
```

------------------------------------------------------------------------

## 9. Stage 6：Dashboard 数据准备

### 目标

把前面阶段的数据组织成适合浏览的树形导航数据。此阶段先准备数据，不限定具体前端技术。

### 主导航结构

``` text
Project
└── User Role
    └── User Journey
        └── User Action
            ├── Business Feature
            ├── Entry Point
            ├── Frontend Entry
            ├── Backend Entry
            ├── Main Files
            ├── Classes / Functions
            ├── Configurations
            └── Evidence / Confidence / Status
```

### Prompt

> 请执行 Stage 6：Dashboard Data Preparation。读取并校验所有阶段产物，建立从 User Role → User Journey → User Action → Business Feature → Code Entry 的可追溯映射。
>
> 主视图必须以用户角色和用户旅程为索引，不得以文件夹或代码目录作为主导航。
>
> 点击用户操作时，优先展示明确的触发入口和主要代码入口。调用关系、依赖图和技术实现细节作为可展开的辅助信息。
>
> 如果入口无法确定，显示 `Unknown / Needs Review`，不得选择一个看起来相关的文件冒充入口。
>
> 对每个展示项保留证据链接、状态和置信度。输出 Dashboard 所需的结构化数据，并检查孤立功能、无代码映射的功能和无业务归属的代码能力。

------------------------------------------------------------------------

## 10. 统一证据与状态规范

### 状态

  状态             含义
  ---------------- ------------------------
  `confirmed`      有直接证据支持
  `inferred`       根据多个证据推断
  `unknown`        当前材料无法判断
  `needs_review`   存在冲突或需要人工确认

### 证据记录建议

``` json
{
  "evidence_id": "EVD-0001",
  "source_type": "source_code",
  "path": "src/example.py",
  "symbol": "parse_document",
  "line_start": 10,
  "line_end": 28,
  "claim": "This function extracts document text.",
  "strength": "direct"
}
```

证据类型可包括：

-   `source_code`
-   `ui`
-   `api`
-   `cli`
-   `test`
-   `documentation`
-   `configuration`
-   `human_confirmation`

置信度可以作为辅助排序信息，但不能替代证据。

------------------------------------------------------------------------

## 11. 验证式执行与人工确认机制

每个阶段都必须遵守：

1.  执行前检查上游产物是否存在、是否完整。
2.  划分有限批次，记录本批处理范围。
3.  执行分析并保存产物。
4.  运行 JSON 格式、ID 唯一性和引用完整性检查。
5.  生成覆盖报告和待确认问题。
6.  暂停，向用户汇报。
7.  用户确认或修正后，再进入下一阶段。

用户确认不意味着必须逐条人工检查全部结果。优先展示高影响、低证据、冲突和未知项目。

### 每批报告模板

``` text
Stage:
Batch:
Input scope:
Files processed:
Files skipped:
Parse failures:
New records:
Updated records:
Unresolved items:
Output files:
Validation result:
Recommended next action:
```

------------------------------------------------------------------------

## 12. 增量更新策略（后续实现）

初版先保证完整扫描和可追溯性。增量更新可以后续实现：

-   通过 Git diff 或文件哈希识别变化；
-   只重新解析变更文件；
-   重新分析受影响的符号和能力；
-   将影响传播到业务功能、旅程和角色；
-   对无法自动判断是否失效的映射标记 `needs_review`；
-   保留上次结果和变更记录。

不能因为文件未变化，就假定它的业务映射永远正确；上游依赖或产品行为变化也可能影响映射。

------------------------------------------------------------------------

## 13. 安全与执行限制

-   默认只读分析仓库。
-   不修改原始业务源码。
-   不执行部署、发布、删除、数据库写入等有副作用操作。
-   不读取、打印或写入密钥、访问令牌、密码和个人敏感数据。
-   不将整个仓库上传到未经用户授权的外部服务。
-   使用外部 LLM 前，说明哪些文件内容会被发送；遵循用户指定的模型和
    API。
-   对生成文件先写入临时文件，验证成功后再替换正式产物。
-   所有路径使用仓库相对路径，避免依赖某台机器的绝对路径。

------------------------------------------------------------------------

## 14. 完成定义（Definition of Done）

### Code Inventory 完成

-   扫描范围内文件均已记录或有跳过原因；
-   代码符号提取可追溯；
-   参数、返回类型缺失时明确标记；
-   解析失败有报告；
-   JSON 结构验证通过。

### Atomic Capabilities 完成

-   候选能力有源码证据；
-   技术能力与业务能力已区分；
-   能力与代码元素的关系可追溯；
-   未知和待验证项没有被隐藏。

### Business Features 完成

-   每项业务功能可追溯到能力；
-   未归类能力保留；
-   不存在仅凭目录名称生成的业务结论。

### User Journeys 完成

-   用户操作与后台自动行为区分；
-   流程顺序有证据或明确标记为推断；
-   每个操作能追溯到业务功能或标记未映射。

### User Roles 完成

-   角色有访问、权限或旅程证据；
-   不凭空添加管理员等常见角色；
-   多角色与多旅程关系得到支持。

### Dashboard 数据完成

-   主导航为角色 → 旅程 → 用户操作；
-   用户操作可映射到代码入口或明确显示未知；
-   每个映射可查看证据和验证状态。

------------------------------------------------------------------------

## 15. Agent 每次启动时的简短入口提示词

当本手册已放入项目目录后，可用以下短提示词启动：

> 请先完整阅读 `.project-intelligence/`
> 中的执行手册和配置文件。不要立即扫描或修改项目。先检查现有分析产物，向我报告当前完成到哪个阶段、哪些批次已完成、哪些问题尚未解决。然后等待我指定本次要执行的阶段或批次。严格遵循手册中的分阶段、分批、持久化、证据和验证规则。未经我明确指示，不得自动进入下一阶段。
