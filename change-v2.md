# Data Agent 变更日志 v2

> **版本**: v2.0  
> **日期**: 2026-05-08  
> **基于**: v1.0 (change-v1.md)  
> **模型**: Qwen3.6-35b-a3b (DashScope API)

---

## 目录

1. [概述](#1-概述)
2. [核心新增: WorkflowPolicy 工作流策略系统](#2-核心新增-workflowpolicy-工作流策略系统)
3. [Agent 推理引擎变更](#3-agent-推理引擎变更)
4. [模型适配器变更](#4-模型适配器变更)
5. [上下文管理变更](#5-上下文管理变更)
6. [工具注册表变更](#6-工具注册表变更)
7. [配置系统变更](#7-配置系统变更)
8. [提示词系统变更](#8-提示词系统变更)
9. [Runner 与 Eval 变更](#9-runner-与-eval-变更)
10. [测试基础设施变更](#10-测试基础设施变更)
11. [Docker 与入口变更](#11-docker-与入口变更)
12. [已知问题与后续方向](#12-已知问题与后续方向)

---

## 1. 概述

本次变更在 v1.0 基础上进行了深度重构，核心目标是解决 v1 中 **40% 任务失败率**（原生 Tool Calling 回退链缺陷）和 **工作流失控**（模型跳过必要步骤直接作答）两大问题。变更范围涵盖：

- **WorkflowPolicy 系统**: 全新的阶段门控工作流引擎，强制 DISCOVER → UNDERSTAND → ANALYZE → VERIFY → SUBMIT 顺序
- **错误分类与智能提示**: 14 种结构化错误类型，针对性的恢复建议
- **JSON 解析重试**: 模型输出无效 JSON 时自动重试一次
- **工具结果缓存**: 避免重复调用只读工具浪费 API
- **配置扩展**: 新增 hard/extreme 超时、workflow_mode、total_time_limit 等参数
- **测试扩展**: test_react.py 从 72 行扩展到 290 行（10 个测试），新增 test_submission_io.py (4 个测试)

---

## 2. 核心新增: WorkflowPolicy 工作流策略系统

### 2.1 概述

**文件**: [agents/react.py](../kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L327-L512)

v1 中模型可以自由选择任何工具，经常跳过 `list_context` 直接执行 SQL 或跳过验证直接 `answer`，导致高失败率。v2 新增 `WorkflowPolicy` 类实现阶段门控工作流。

### 2.2 三种运行模式

| 模式 | 行为 |
|------|------|
| `off` | 完全禁用工作流策略，等同 v1 行为 |
| `soft` (默认) | 阻止违规操作，返回被阻止的观察 + 建议下一步动作，但不强制重写 |
| `strict` | 阻止违规操作，并在提示中**强制**推荐下一步动作名称 |

通过 `ReActAgentConfig.workflow_mode` 或环境变量 `WORKFLOW_MODE` 配置。

### 2.3 五阶段工作流

```
DISCOVER → UNDERSTAND → ANALYZE → VERIFY → SUBMIT
```

**阶段定义**:

| 阶段 | 触发条件 | 允许的工具 |
|------|----------|-----------|
| DISCOVER | 尚未成功调用 `list_context` | `list_context`, `get_doc_info`, `read_csv`, `read_doc`, `read_doc_segment`, `read_json`, `search_doc_keywords`, `summarize_csv`, `summarize_sqlite`, `inspect_sqlite_schema` |
| UNDERSTAND | 已 discover 但尚未理解和摘要数据源 | DISCOVER 工具 + `execute_python` |
| ANALYZE | 已理解数据但尚未成功分析 | `execute_context_sql`, `execute_python`, `read_csv`, `read_json`, `read_doc_segment`, `search_doc_keywords`, `summarize_csv`, `summarize_sqlite`, `inspect_sqlite_schema` |
| VERIFY | 已有成功分析但尚未独立验证 | 分析工具 + `answer` |
| SUBMIT | 所有条件满足 | `answer` + 验证工具 |

### 2.4 阶段判定逻辑

`WorkflowPolicy.stage()` ([L381-392](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L381-L392)) 按优先级检查：
1. 当前 action 是否为 `answer` → SUBMIT
2. 是否成功调用过 `list_context` → 否则 DISCOVER
3. 是否有数据理解步骤（summarize/inspect/read） → 否则 UNDERSTAND
4. 是否有非空分析结果 → 否则 ANALYZE
5. 是否满足验证条件（2+ 分析步骤或有 verified 标记） → 否则 VERIFY
6. 以上全部满足 → SUBMIT

### 2.5 违规检测与阻止

`WorkflowPolicy.decide()` ([L334-379](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L334-L379)) 按优先级检查以下违规：

| 优先级 | 检查项 | 违规时返回的错误类型 | 建议重写动作 |
|--------|--------|---------------------|-------------|
| 1 | 重复失败调用（相同 action + input） | `repeated_failure` | `list_context` |
| 2 | 未 discover 就分析 | `discover_required` | `list_context` |
| 3 | Python 连续失败且无数据理解 | `python_recovery_required` | `summarize_csv` |
| 4 | hard/extreme 任务尝试读取超长文档 | `document_workflow_required` | `search_doc_keywords` |
| 5 | SQL 执行前未检查 schema | `schema_required` | `inspect_sqlite_schema` |
| 6 | Python 执行前无数据理解 | `understand_required` | `summarize_csv` |
| 7 | answer 前未验证 | `verification_required` | `execute_python` |
| 8 | 工具不在当前阶段允许列表中 | `workflow_stage_violation` | 推荐当前阶段工具 |

### 2.6 新增数据结构

#### `PolicyDecision` ([L130-136](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L130-L136))

```python
@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str
    hint: str
    stage: str
    rewritten_action: str | None = None
```

#### `ERROR_HINTS` ([L25-74](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L25-L74))

新增 14 种结构化错误类型，每种包含具体的中文/英文恢复指导：

| 错误类型 | 触发场景 |
|----------|----------|
| `sqlite_error` | SQL 执行失败 |
| `empty_result` | 查询返回零行 |
| `python_error` | Python 执行失败 |
| `file_not_found` | 文件路径不存在 |
| `timeout` | 执行超时 |
| `key_error` | 列名/键不存在 |
| `schema_required` | SQL 前未检查 schema |
| `discover_required` | 未调用 list_context |
| `understand_required` | 分析前未理解数据 |
| `verification_required` | answer 前未验证 |
| `repeated_failure` | 重复失败调用 |
| `python_recovery_required` | Python 连续失败且无数据检查 |
| `document_workflow_required` | hard/extreme 任务读长文档 |
| `tool_input_error` | 工具输入无效 |

### 2.7 工具分类常量

新增工具集合常量 ([L76-105](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L76-L105)):

```python
SQL_SCHEMA_TOOLS = {"inspect_sqlite_schema", "summarize_sqlite"}
DISCOVER_TOOLS = {...}   # 9 个工具
UNDERSTAND_TOOLS = DISCOVER_TOOLS | {"execute_python"}  # 10 个工具
ANALYZE_TOOLS = {...}    # 9 个工具
VERIFY_TOOLS = {...}     # 6 个工具
SUBMIT_TOOLS = {"answer"}
```

`OPTIONAL_TOOL_FIELDS` ([L107-114](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L107-L114)) 标记每个工具的非必填参数，用于原生 TC schema 生成时区分 required/optional。

---

## 3. Agent 推理引擎变更

### 3.1 ReActAgentConfig 扩展

**文件**: [agents/react.py#L117-127](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L117-L127)

**新增字段**:

```python
workflow_mode: str = "soft"  # 新增: off | soft | strict
```

### 3.2 JSON 解析重试机制

**新增方法 `_complete_text_step_with_retry()`** ([L543-550](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L543-L550)):

v1 中 JSON 解析失败直接抛出异常。v2 中解析失败时自动注入错误提示消息并重试一次：

```python
def _complete_text_step_with_retry(self, messages):
    raw_response = self.model.complete(messages)
    try:
        return raw_response, parse_model_step(raw_response)
    except Exception as exc:
        retry_messages = [*messages, _retry_prompt_for_invalid_model_step(exc)]
        retry_raw_response = self.model.complete(retry_messages)
        return retry_raw_response, parse_model_step(retry_raw_response)
```

**新增函数 `_retry_prompt_for_invalid_model_step()`** ([L244-252](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L244-L252)): 向模型注入 "Your previous tool call/action was invalid" 提示。

### 3.3 Schema Guard 观察

**新增方法 `_schema_guard_observation()`** ([L579-598](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L579-L598)):

在工具实际执行前检查 SQL 操作是否已先检查 schema。仅在 `workflow_mode != "off"` 时生效。如果未检查 schema 就尝试 SQL，直接返回包含错误提示的观察，**不实际执行工具**。

### 3.4 观察增强

**新增方法 `_augment_observation()`** ([L600-615](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L600-L615)):

每个工具观察在执行后统一增强：
- 失败的观察添加 `error_type` 和 `hint` 字段
- SQL 查询返回零行时添加 `warning_type: "empty_result"` 和相应提示
- Policy 阻止的观察保留 `policy_reason` 作为 `error_type`

### 3.5 Policy 观察构建

**新增方法 `_policy_observation()`** ([L617-633](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L617-L633)):

当 WorkflowPolicy 阻止操作时，构建包含以下信息的观察：
- `error_type`: 违规原因
- `policy_stage`: 当前工作流阶段
- `rewritten_action`: 建议的下一步动作
- `blocked_action` / `blocked_action_input`: 被阻止的动作

### 3.6 run() 方法重构

`run()` 方法 ([L635-737](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L635-L737)) 主要变更：

1. **实例化 WorkflowPolicy** (L529): `self.workflow_policy = WorkflowPolicy(mode=self.config.workflow_mode)`
2. **初始化 ContextManager** (L528): 接受外部传入或自动创建
3. **JSON 重试** (L653, L663, L668, L671): 所有文本补全路径均使用 `_complete_text_step_with_retry`
4. **Policy 决定** (L683-686): 每一步都先检查 WorkflowPolicy，除非是超时强制 answer
5. **Schema Guard** (L689-690): Policy 通过后，SQL 操作还需通过 schema guard
6. **Obs 增强** (L703): 所有观察统一经过 `_augment_observation` 处理
7. **Native TC 异常处理改进** (L665-669): 异常时注入重试提示消息

### 3.7 新增辅助函数

| 函数 | 行号 | 用途 |
|------|------|------|
| `_stringify_error_payload()` | L255-261 | 将错误 payload 统一转为字符串 |
| `_classify_error()` | L264-280 | 基于 action 名 + payload 文本分类错误类型 |
| `_error_hint()` | L283-285 | 返回 (error_type, hint) 元组 |
| `_is_tool_step()` | L288-289 | 判断步骤是否为实际工具调用（排除内部操作） |
| `_successful_steps()` | L292-297 | 过滤成功步骤，可选按 action 过滤 |
| `_has_successful_action()` | L300-301 | 检查是否有成功的指定 action |
| `_failed_signature()` | L304-310 | 生成失败步骤的签名（action + input） |
| `_model_step_signature()` | L313-319 | 生成模型步骤的签名 |
| `_step_has_empty_result()` | L322-324 | 检查 SQL 步骤是否返回空结果 |

---

## 4. 模型适配器变更

### 4.1 OpenAIModelAdapter.complete() 重构

**文件**: [agents/model.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/model.py)

v1 中 `complete()` 有独立的 API 调用逻辑。v2 中简化为委托调用：

```python
def complete(self, messages: list[ModelMessage]) -> str:
    content, _, _ = self.complete_with_tools(messages, tools=None, tool_choice="none")
    return content
```

**变更原因**: 消除代码重复，统一 API 调用路径。

### 4.2 complete_with_tools() 改进

- `tool_choice` 参数现在由调用方传入（v1 中硬编码为 `"auto"`）
- `tool_choice="none"` 时强制模型不使用工具（用于 `complete()` 委托）
- 工具调用响应中正确保留 `id` 字段 ([L119](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/model.py#L119))

---

## 5. 上下文管理变更

### 5.1 ContextManager 扩展

**文件**: [context/manager.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/context/manager.py)

**新增功能**:

1. **`add_assistant()` 支持 tool_calls 参数** ([L18-19](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/context/manager.py#L18-L19)):
   ```python
   def add_assistant(self, content: str, tool_calls: list[dict] | None = None) -> None:
   ```
   支持原生 TC 对话轮次的 assistant 消息（包含 tool_calls 字段）。

2. **新增 `add_tool_result()` 方法** ([L21-22](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/context/manager.py#L21-L22)):
   ```python
   def add_tool_result(self, tool_call_id: str, content: str) -> None:
   ```
   为原生 TC 流程中 `role="tool"` 的消息提供支持。

---

## 6. 工具注册表变更

### 6.1 工具结果缓存

**文件**: [tools/registry.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/tools/registry.py)

**新增功能**: 避免相同参数下重复调用只读工具浪费 API token。

```python
CACHEABLE_TOOLS = frozenset({
    "get_doc_info", "inspect_sqlite_schema", "list_context",
    "read_csv", "read_doc", "read_doc_segment", "read_json",
    "search_doc_keywords", "summarize_csv", "summarize_sqlite",
})
```

**ToolRegistry 新增字段** ([L176-182](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/tools/registry.py#L176-L182)):
- `cache_enabled: bool = True`
- `cacheable_tools: frozenset[str] = CACHEABLE_TOOLS`
- `_cache: dict[str, ToolExecutionResult]`

**缓存键生成** `_cache_key()` ([L60-62](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/tools/registry.py#L60-L62)): 基于 `context_dir::action::normalized_input` 三元组。

**execute() 方法更新** ([L192-206](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/tools/registry.py#L192-L206)): 执行前检查缓存，成功且非终端的只读工具结果自动缓存（deepcopy 防止引用污染）。

### 6.2 Native TC 解析增强

`_parse_native_tool_calls()` ([L220-241](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L220-L241)) 新增 `valid_tool_names` 参数，验证返回的工具名是否在已知工具列表中。

---

## 7. 配置系统变更

### 7.1 AgentConfig 新增字段

**文件**: [config.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/config.py)

```python
workflow_mode: str = "soft"  # L34 - 新增
```

### 7.2 RunConfig 新增字段

```python
task_timeout_seconds: int = 600   # L44 - 新增: 通用任务超时
timeout_hard: int = 300           # L47 - 新增: hard 任务超时
timeout_extreme: int = 480        # L48 - 新增: extreme 任务超时
total_time_limit: int = 43200     # L49 - 新增: 12 小时总时限
```

v1 中只有 `timeout_easy` 和 `timeout_medium`，v2 补全了全部四个难度级别的超时配置。

### 7.3 环境变量配置扩展

`load_app_config_from_env()` ([L127-146](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/config.py#L127-L146)):

| 环境变量 | 默认值 | 用途 |
|----------|--------|------|
| `WORKFLOW_MODE` | `soft` | 工作流策略模式 |
| `MAX_WORKERS` | `8` | 并发 worker 数 |
| `TASK_TIMEOUT_SECONDS` | `600` | 通用任务超时 |
| `MODEL_NAME` | `qwen3.5-35b-a3b` | 模型名称（注意默认从 v1 的 `qwen3.6-35b-a3b` 改为 `qwen3.5-35b-a3b`） |

### 7.4 dev.yaml 更新

**文件**: [configs/dev.yaml](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/configs/dev.yaml)

新增字段与 `config.py` 默认值保持同步：`timeout_hard: 300`, `timeout_extreme: 480`, `total_time_limit: 43200`。

---

## 8. 提示词系统变更

### 8.1 REACT_SYSTEM_PROMPT 增强

**文件**: [agents/prompt.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/prompt.py)

**新增 "Sequential Workflow" 章节** ([L50-56](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/prompt.py#L50-L56)):

```
## Sequential Workflow
Follow this order unless a tool observation tells you to recover differently:
1. DISCOVER: call `list_context`.
2. UNDERSTAND: summarize or inspect the relevant data source.
3. ANALYZE: run SQL/Python calculations.
4. VERIFY: cross-check the result with an independent SQL/Python check.
5. SUBMIT: call `answer`.
```

**新增 Rule #9** ([L67](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/prompt.py#L67)): 强调 `answer` 工具的 columns 和 rows 格式约束。

### 8.2 build_system_prompt() 参数扩展

[L101-110](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/prompt.py#L101-L110):

- 新增 `system_prompt` 可选参数，支持外部注入自定义系统提示词
- 始终追加 `RESPONSE_EXAMPLES`（v1 中示例仅附加到工具描述中）

---

## 9. Runner 与 Eval 变更

### 9.1 Runner 配置传递

**文件**: [run/runner.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/run/runner.py)

- `_run_single_task_core()` 现在通过 `ReActAgentConfig` 传递 `max_steps` 和 `workflow_mode` 给 Agent ([L121-124](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/run/runner.py#L121-L124))
- `_run_single_task_with_timeout()` 支持全部四个难度级别的超时映射 ([L152-158](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/run/runner.py#L152-L158))
- `run_benchmark()` 支持通过参数注入共享 `model` 和 `tools` 实例（用于测试场景） ([L240-241](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/run/runner.py#L240-L241))
- 单 worker 模式使用共享实例以避免重复初始化 ([L269-270](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/run/runner.py#L269-L270))

### 9.2 Eval 地真值搜索扩展

**文件**: [eval/run_eval.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/eval/run_eval.py)

`_find_ground_truth()` 现在搜索 6 个候选路径 ([L18-30](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/eval/run_eval.py#L18-L30))，包括官方竞赛输出目录布局 `data/public/output/{task_id}/gold.csv`。

### 9.3 CLI 改进

**文件**: [cli.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/cli.py)

- `run_benchmark_command` 现在支持 `--limit` 参数限制任务数量
- 进度条显示改进：紧凑模式显示 ok/fail/run/queue 计数 + 速率 + 最后完成的任务

---

## 10. 测试基础设施变更

### 10.1 test_react.py 大幅扩展

**文件**: [tests/test_react.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/test_react.py)

从 v1 的 72 行扩展到 290 行，测试数从 5 个增加到 10 个：

| 测试 | 类型 | 说明 |
|------|------|------|
| `test_parse_valid_json` | 保留 | JSON 解析（带 fence） |
| `test_parse_json_without_fence` | 保留 | JSON 解析（无 fence） |
| `test_parse_json_invalid_raises` | 保留 | 非法 JSON 抛出异常 |
| `test_force_answer_on_timeout` | 修改 | 超时强制 answer（新增 `workflow_mode="off"`） |
| `test_max_consecutive_errors_intervention` | 修改 | 连续错误干预（禁用 planning/reflection） |
| `test_sql_requires_schema_before_execution` | **新增** | Schema guard 测试 |
| `test_workflow_requires_list_context_before_analysis` | **新增** | DISCOVER 阶段门控测试 |
| `test_workflow_blocks_premature_answer_without_verification` | **新增** | VERIFY 阶段门控测试 |
| `test_workflow_allows_sequential_success_path` | **新增** | 完整工作流成功路径测试 |
| `test_workflow_blocks_repeated_failed_call` | **新增** | 重复失败调用检测测试 |
| `test_bad_native_tool_call_falls_back_to_text_retry` | **新增** | 原生 TC 空 action 回退重试 |
| `test_tool_registry_caches_read_only_tools` | **新增** | 工具结果缓存测试 |

新增 `_make_policy_tools()` 辅助函数 ([L47-88](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/test_react.py#L47-L88))，创建专门用于 workflow policy 测试的内嵌工具注册表。新增 `BadNativeToolCallAdapter` 类 ([L238-249](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/test_react.py#L238-L249)) 模拟原生 TC 返回空 function name 的场景。

### 10.2 新增 test_submission_io.py

**文件**: [tests/test_submission_io.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/test_submission_io.py) (97 行)

4 个新测试，验证竞赛 Docker 环境下的 I/O 行为：

| 测试 | 说明 |
|------|------|
| `test_env_config_defaults_to_official_io` | 验证 `load_app_config_from_env` 默认 `/input` → `/output` 路径 |
| `test_official_output_layout_has_no_nested_run_id` | 验证 `nest_run_id=False` 时输出目录布局正确 |
| `test_read_csv_preview_streams_preview_rows_but_counts_all_rows` | 验证 `read_csv_preview` 返回 `truncated` 标记和 `row_count` |
| `test_find_ground_truth_supports_official_gold_layout` | 验证 `_find_ground_truth` 支持官方 `data/public/output/` 布局 |

### 10.3 perf_test.py 小幅修改

- `run_performance_test()` 中 Agent 现在使用 `ReActAgentConfig(max_steps=config.agent.max_steps)` ([L175](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/perf_test.py#L175))
- 瓶颈分析中 Unicode 字符已替换为 ASCII（v1 中已修复）

### 10.4 gen_report.py 小幅修改

- 任务失败时现在输出更详细的根因分析（包含 Qwen3.5 native TC 行为说明）
- 建议章节更新为 P0/P1/P2 优先级标记

---

## 11. Docker 与入口变更

### 11.1 Dockerfile 更新

**文件**: [Dockerfile](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/Dockerfile)

- 使用 `uv sync --frozen --no-dev` 替代原始安装方式
- 创建 `/logs` 目录
- 使用 `entrypoint.sh` 作为入口点

### 11.2 entrypoint.sh 新增

**文件**: [entrypoint.sh](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/entrypoint.sh)

Docker 入口脚本，创建 `/output` 和 `/logs` 目录后运行 `main.py`，输出通过 `tee` 同时写入日志和控制台。

### 11.3 main.py 健壮性改进

**文件**: [main.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/main.py)

- 新增 API 配置缺失检查 ([L29-31](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/main.py#L29-L31)): 缺少 `MODEL_API_URL` 或 `MODEL_API_KEY` 时打印错误并退出
- 创建 `/logs` 和 `/output` 目录 ([L12, L38](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/main.py#L12))
- 日志同时输出到 stdout 和 `/logs/main.log` ([L18-20](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/main.py#L18-L20))

### 11.4 submission 目录新增

- `submission/SUBMISSION.md` (1293 bytes) - 提交说明文档
- `submission/team1199_v1.tar.gz` (233MB) - 提交包

---

## 12. 已知问题与后续方向

### 12.1 P0 — WorkflowPolicy 在 strict 模式下的行为验证

**现象**: strict 模式的理论设计完整，但尚未在真实竞赛任务上进行充分的端到端测试。  
**建议方案**: 在 Docker 环境中用 strict 模式运行完整任务集，对比 soft 模式的表现差异。

### 12.2 P1 — Native TC 重试链仍有缺陷

**现象**: 新增的 `_complete_text_step_with_retry` 只重试一次。如果模型在重试后仍产出无效 JSON，任务仍会失败。  
**建议方案**: 考虑在多次失败后引导模型切换策略（如仅使用简单工具），而非仅仅重试 JSON 解析。

### 12.3 P1 — 工作流测试覆盖不完整

**现象**: 当前 12 个 workflow 测试覆盖了核心门控逻辑，但未覆盖以下场景：
- `strict` 模式的强制重写行为
- hard/extreme 文档工作流限制
- `workflow_mode="off"` 的完全放行验证  
**建议方案**: 补充上述三个场景的测试。

### 12.4 P2 — ContextManager 当前未在工具结果中使用

**现象**: ContextManager 的 `add_tool_result` 方法已定义，但当前 `_build_messages` 中仍以 user 角色注入观察（而非 tool 角色）。  
**建议方案**: 在原生 TC 流程中使用真正的 `role="tool"` 消息，以获得更好的模型行为。

### 12.5 P2 — 性能基准更新

**现象**: v1 的性能测试报告显示 60% 成功率（3/5），v2 引入 WorkflowPolicy 后理论上成功率应显著提升，但尚未重新运行性能基准。  
**建议方案**: 使用相同的 5 个 easy 任务重新运行 `perf_test.py`，更新 `performance_report.json`。

---

## 附录 A: 文件变更清单

### 新增文件 (3 个)

| 文件路径 | 行数 | 类型 |
|----------|------|------|
| `tests/test_submission_io.py` | 97 | 提交 I/O 测试 |
| `entrypoint.sh` | 16 | Docker 入口脚本 |
| `submission/SUBMISSION.md` | ~25 | 提交文档 |

### 修改文件 (12 个)

| 文件路径 | 变更类型 | 主要变更内容 |
|----------|----------|-------------|
| `src/data_agent_baseline/agents/react.py` | 重大重构 | 新增 WorkflowPolicy (185行)、ERROR_HINTS、JSON 重试、Schema Guard、Obs 增强；run() 主线重构 |
| `src/data_agent_baseline/agents/model.py` | 重构 | complete() 委托给 complete_with_tools；tool_choice 参数化 |
| `src/data_agent_baseline/agents/prompt.py` | 扩展 | 新增 Sequential Workflow 章节、Rule #9 |
| `src/data_agent_baseline/agents/runtime.py` | 无变化 | — |
| `src/data_agent_baseline/context/manager.py` | 扩展 | add_assistant 支持 tool_calls；新增 add_tool_result |
| `src/data_agent_baseline/tools/registry.py` | 扩展 | 工具结果缓存（CACHEABLE_TOOLS、_cache）、execute() 缓存逻辑 |
| `src/data_agent_baseline/config.py` | 扩展 | workflow_mode、timeout_hard/extreme、total_time_limit、环境变量 |
| `src/data_agent_baseline/run/runner.py` | 扩展 | workflow_mode 传递、全难度超时、共享 model/tools 注入 |
| `src/data_agent_baseline/eval/run_eval.py` | 扩展 | _find_ground_truth 6 路径搜索 |
| `src/data_agent_baseline/cli.py` | 改进 | --limit 参数、紧凑进度显示 |
| `main.py` | 改进 | API 配置检查、/logs 目录、双输出日志 |
| `tests/test_react.py` | 重大扩展 | 72→290 行，5→10 测试，新增 workflow policy 和 native TC 回退测试 |
| `tests/perf_test.py` | 微调 | Agent 配置使用 ReActAgentConfig |
| `tests/gen_report.py` | 微调 | 详细根因分析、P0/P1/P2 优先级 |
| `configs/dev.yaml` | 同步 | 新增 timeout_hard/extreme、total_time_limit |
| `Dockerfile` | 更新 | uv sync、/logs、entrypoint.sh |

---

## 附录 B: v1 已知问题的解决状态

| v1 问题 | 状态 | v2 解决方案 |
|----------|------|------------|
| P0: 原生 TC 回退链缺陷 | **部分解决** | JSON 解析重试机制 + 增强的 Native TC 异常回退 + BadNativeToolCallAdapter 测试 |
| P1: 为 easy 任务禁用规划 | **未解决** | 待后续版本 |
| P1: 利用闲置 CPU 资源 | **配置已更新** | `max_workers` 默认值 8（config.py L43），可通过 env `MAX_WORKERS` 覆盖 |
| P2: Token 使用量监控 | **未解决** | `_total_tokens` 仍只追踪未输出 |
| P2: Docker 环境验证 | **改进** | `test_submission_io.py` 新增官方布局测试；Dockerfile 和 entrypoint.sh 完善 |

---

## 附录 C: WorkflowPolicy 决策流程图

```
model_step → WorkflowPolicy.decide()
  ├─ action 是内部操作？ → 放行
  ├─ 重复失败调用？ → 阻止 (repeated_failure)
  ├─ 未 discover 且非 list_context？ → 阻止 (discover_required)
  ├─ Python 连续失败且无数据理解？ → 阻止 (python_recovery_required)
  ├─ hard/extreme 读长文档？ → 阻止 (document_workflow_required)
  ├─ SQL 未先查 schema？ → 阻止 (schema_required)
  ├─ Python 无数据理解？ → 阻止 (understand_required)
  ├─ answer 未验证？ → 阻止 (verification_required)
  ├─ 工具不在当前阶段？ → 阻止 (workflow_stage_violation)
  └─ 全部通过 → 放行
```
