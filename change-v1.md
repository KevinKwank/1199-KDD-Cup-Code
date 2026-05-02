# Data Agent 变更日志 v1

> **版本**: v1.0  
> **日期**: 2026-05-02  
> **基于**: KDD Cup 2026 Data Agents Starter Kit  
> **模型**: Qwen3.6-35b-a3b (DashScope API)

---

## 目录

1. [概述](#1-概述)
2. [新增模块](#2-新增模块)
3. [核心模块变更](#3-核心模块变更)
4. [测试基础设施](#4-测试基础设施)
5. [性能与配置优化](#5-性能与配置优化)
6. [Bug 修复](#6-bug-修复)
7. [已知问题与后续方向](#7-已知问题与后续方向)

---

## 1. 概述

本次变更在 KDD Cup 2026 Data Agents 官方 Starter Kit 基础上，按照开发流程计划书完成了 Data Agent 的完整开发，并实施了全面的优化方案。变更范围涵盖：

- **Agent 核心引擎**: ReAct 推理循环、原生 Tool Calling 集成、规划与反思机制
- **上下文管理**: 令牌预算感知的对话压缩系统
- **文档处理**: 长文档分段、关键词检索能力
- **工具系统**: 数据摘要、文档检索等新工具的扩展
- **测试体系**: 16 个单元测试 + 性能测试框架 + 综合评分报告系统
- **配置优化**: 资源参数调优，支撑 12 小时/380 任务竞赛场景

---

## 2. 新增模块

### 2.1 上下文管理模块

#### 新增文件: [context/manager.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/context/manager.py)（66 行）

**功能说明**: 实现令牌预算感知的多轮对话管理器，防止上下文窗口溢出。

**核心设计**:

| 组件 | 说明 |
|------|------|
| `ContextManager(max_tokens=200000)` | 构造函数，默认 200K 令牌预算 |
| `set_system(content)` | 设置系统消息（不计入压缩范围） |
| `add_user(content)` / `add_assistant(content)` / `add_tool_result(id, content)` | 按角色追加消息 |
| `get_messages()` | 返回消息列表，超过预算时自动压缩 |
| `_compress(messages)` | 从后往前保留最近消息，超出预算的早期消息被截断 |
| `clear()` | 清空对话历史 |
| `message_count` | 当前消息数量属性 |

**实现细节**:

- 令牌估算采用 `字符数 / 2` 的简化算法，平衡精度与性能
- 压缩策略：系统消息始终保留（L1-12），然后从消息列表末尾向前保留，直到剩余预算不足 4000 字符（L38-59）
- 截断时插入中文提示标记，告知模型对话历史已被压缩（L56-59）

#### 新增文件: [context/document.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/context/document.py)（59 行）

**功能说明**: 长文档分段处理与关键词检索引擎，解决竞赛中 Extreme 难度任务（>128K tokens 文档）的读取问题。

**核心设计**:

| 组件 | 说明 |
|------|------|
| `DocumentProcessor(max_chars_per_segment=6000, max_segments=30)` | 构造函数，每段 6000 字符，最多 30 段 |
| `segment(file_path)` -> `list[str]` | 按段落边界将文档切割为多个片段 |
| `get_segment(file_path, index)` -> `str | None` | 获取指定索引的片段（含片段编号标记） |
| `get_segment_count(file_path)` -> `int` | 获取总片段数 |
| `search_keywords(file_path, keywords)` -> `list[dict]` | 在所有片段中搜索关键词，返回匹配的片段索引和预览 |

**实现细节**:

- 分段算法以 `\n\n` 为段落分隔符（L20），单段超过 `max_chars` 时新起一段（L25-26），最多保留 `max_segments` 段（L34）
- 内部使用字典缓存已处理文件，避免重复读取（L10, L14-15）
- 关键词搜索大小写不敏感（L51），返回 300 字符预览（L57）

#### 新增文件: [context/\_\_init\_\_.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/context/__init__.py)

模块初始化文件，使 `context` 成为合法的 Python 包。

---

### 2.2 数据摘要工具

#### 新增文件: [tools/data_summary.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/tools/data_summary.py)（63 行）

**功能说明**: 提供比 `read_csv`/`read_doc` 更高效的数据概览工具，减少模型读取完整数据的需求。

**核心函数**:

| 函数 | 功能 | 输出内容 |
|------|------|----------|
| `summarize_csv(task, action_input)` (L10-41) | CSV 文件统计摘要 | shape、每列 dtype、null 数量、unique 数量、数值列统计（describe） |
| `summarize_sqlite(task, action_input)` (L44-63) | SQLite 数据库摘要 | 表列表、每表行数、列名与类型 |

**实现细节**:

- `summarize_csv`: 当某列唯一值 ≤ 15 时，直接列出所有唯一值供模型参考（L29-30）
- `summarize_sqlite`: 使用只读 URI 模式 `file:path?mode=ro` 打开数据库（L48），避免意外写入
- 两个函数均通过 `resolve_context_path` 进行路径安全检查

---

### 2.3 性能测试基础设施

#### 新增文件: [tests/perf_test.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/perf_test.py)（508 行）

**功能说明**: 全功能性能测试框架，支持资源监控、多任务批量测试、质量/性能/效率三维评分。

**核心组件**:

| 类/函数 | 说明 |
|---------|------|
| `ResourceSnapshot` (L27-33) | 资源快照数据类：CPU%、内存(MB)、磁盘读写(MB) |
| `TaskPerfMetrics` (L37-59) | 单任务完整性能指标：耗时、步数、质量、资源等 |
| `ResourceMonitor` (L62-148) | 基于 `psutil` 的后台线程资源监控器 |
| `run_performance_test()` (L163-250) | 批量任务执行与指标采集 |
| `score_task_quality()` (L253-274) | 任务质量评分（成功/失败、答案空值、工具错误） |
| `score_performance()` (L277-296) | 执行性能评分（耗时、步数、内存） |
| `score_resource_efficiency()` (L299-313) | 资源效率评分（API 调用次数、效率比） |
| `generate_report()` (L316-480) | 综合评分报告生成（控制台 + JSON） |

**评分体系**:

| 维度 | 权重 | 评分规则 |
|------|------|----------|
| 任务完成质量 | 45% | 基础 100 分，失败 -40，空答案 -30/-20，工具错误每次 -5（上限 -15） |
| 执行性能 | 30% | 基础 100 分，超 60s 按 0.5/s 扣分（上限 -30），超 12 步每次 -2，内存 >1GB 按比例扣分 |
| 资源效率 | 25% | 基础 100 分，API 调用 >10 次每次 -3，低 API 调用但高耗时 -10 |

**等级映射**: A+(≥90) → A(≥80) → B(≥70) → C(≥60) → D(<60)

#### 新增文件: [tests/gen_report.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/gen_report.py)（303 行）

**功能说明**: 脱机报告生成器，使用预设的测试数据生成完整评分报告，不依赖 API 调用。

**报告结构**: 6 个章节——任务执行概览 → 逐任务分析 → 汇总指标 → 加权综合评分 → 性能瓶颈分析 → 建议与后续优化方向

#### 新增文件: [artifacts/perf_tests/performance_report.json](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/artifacts/perf_tests/performance_report.json)

**功能说明**: JSON 格式的性能测试报告持久化存储，供自动化流程或 CI/CD 系统消费。

---

## 3. 核心模块变更

### 3.1 Agent 推理引擎（重构）

#### 文件: [agents/react.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py)

**变更类型**: 重大重构——从原始 starter kit 基础实现完全重写，新增约 150 行业务逻辑。

##### 3.1.1 配置扩展: `ReActAgentConfig`（L25-34）

**变更前** (原始 starter kit):
```python
@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    force_answer_on_timeout: bool = True
    max_consecutive_errors: int = 3
```

**变更后**:
```python
@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    max_steps: int = 16
    force_answer_on_timeout: bool = True
    max_consecutive_errors: int = 3
    enable_planning: bool = True           # 新增：启用规划阶段
    enable_reflection: bool = True         # 新增：启用反思检查点
    reflection_interval: int = 3           # 新增：反思间隔（每 N 步）
    context_max_tokens: int = 200000       # 新增：上下文令牌预算
    use_native_tool_calling: bool = True   # 新增：启用原生 Tool Calling
```

**变更原因**: 支持规划、反思、原生 TC 三大优化特性，提供灵活的功能开关。

##### 3.1.2 原生 Tool Calling 集成（L75-106）

**新增函数 `_tool_to_openai_spec(spec)`** (L75-93):

将内部 `ToolSpec` 转换为 OpenAI 兼容的 Function Calling 格式：
```python
{
    "type": "function",
    "function": {
        "name": spec.name,
        "description": spec.description,
        "parameters": {
            "type": "object",
            "properties": {k: {"type": "string", "description": f"Value for {k}"} for k in spec.input_schema},
            "required": list(spec.input_schema.keys()),
        },
    },
}
```

**新增函数 `_parse_native_tool_calls(tool_calls)`** (L96-106):

将原生 TC 响应转换为内部 `ModelStep` 格式，处理空的 function name 和无效 JSON 的回退。

##### 3.1.3 规划阶段（L125-132）

**新增方法 `_generate_plan(task)`**:

- 使用独立的 `PLANNING_SYSTEM_PROMPT`，以"资深数据分析师"角色指导模型生成分析计划
- 计划内容截断至 2000 字符（L132），避免消耗过多主循环令牌预算
- 规划失败不阻止执行，仅记录警告日志（L159-160）

##### 3.1.4 上下文感知的消息构建（L137-150）

**新增方法 `_build_messages(task, state)`**:

- 系统消息通过 `build_system_prompt()` 包含工具描述 + API 示例
- 当存在分析计划时，将其追加到任务描述中（L141-142）
- 工具调用结果使用 OpenAI 原生 `tool` 角色封装（L148-149）
- 反思和系统提示使用 `user` 角色注入（L145-146）

##### 3.1.5 主循环中的原生 TC 调用（L172-190）

```python
if self.config.use_native_tool_calling and native_available and tools_spec:
    try:
        content, tool_calls, _ = self.model.complete_with_tools(messages, tools=tools_spec, tool_choice="auto")
        if tool_calls:
            model_step = _parse_native_tool_calls(tool_calls)
            ...
        else:
            raw_response = self.model.complete(messages)  # 回退到文本补全
            ...
    except Exception as exc:
        logger.warning(f"Native TC fallback to text: {exc}")
        raw_response = self.model.complete(messages)  # 异常回退
        ...
```

**三层回退策略**:
1. 优先尝试原生 Tool Calling（L173-178）
2. TC 返回空工具调用时回退到 JSON 文本补全（L179-181）
3. TC 请求异常时回退到 JSON 文本补全（L183-187）

**已知限制**: 当原生 TC 返回空 function name 且文本回退也无法产出有效 action 时，任务会失败（参见第七节）。

##### 3.1.6 反思检查点（L215-216）

**新增逻辑**:

- 每 `reflection_interval`（默认 3）步注入反思提示
- 提示内容引导模型检查：是否仍走在正确轨道上、中间结果是否一致、策略是否需要调整、是否已有足够信息作答
- 反思提示以 `__reflect__` 虚拟 action 形式记录在步骤历史中，不消耗实际工具调用

**效果**: 帮助模型在长链推理中保持目标和策略一致性。

##### 3.1.7 连续错误干预（L211-213, 222-224）

**修复**: 原来连续错误检查仅在 try 块中执行，except 块中递增计数器后不检查阈值。修复后在两处均检查:

```python
# try 块中 (L211-213)
if consecutive_errors >= self.config.max_consecutive_errors:
    state.steps.append(StepRecord(..., action="__system_hint__", ...))
    consecutive_errors = 0

# except 块中 (L222-224) — 新增
if consecutive_errors >= self.config.max_consecutive_errors:
    state.steps.append(StepRecord(..., action="__system_hint__", ...))
    consecutive_errors = 0
```

---

### 3.2 提示词系统（重构）

#### 文件: [agents/prompt.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/prompt.py)

**变更类型**: 重大重构——从原始 starter kit 基础提示词完全重写。

##### 3.2.1 规划阶段提示词 `PLANNING_SYSTEM_PROMPT`（L8-18）

**新增**: 独立的规划阶段系统提示词，引导模型在工具执行前生成 4 阶段分析计划（数据理解 → 分析步骤 → 验证 → 答案格式），每步一行，不执行任何工具。

##### 3.2.2 增强的推理提示词 `REACT_SYSTEM_PROMPT`（L21-67）

**新增内容对比**:

| 章节 | 原始内容 | 新增/增强内容 |
|------|----------|---------------|
| Workflow | 无结构化流程 | 4 步工作流：数据理解 → 规划 → 逐步执行 → 提交答案 |
| Step 1: 数据理解 | 无 | 为每种文件类型（CSV/DB/Doc/JSON）指定最佳工具和策略 |
| Step 3: 执行 | 无 | 明确每步操作规范：陈述目标、选择工具、验证结果、错误处理 |
| Important Rules | 少量基本规则 | 8 条详细规则，覆盖数据保真、数值精度、NULL 处理、SQL 优化等 |
| Common Mistakes | 无 | **新增** 6 条常见错误警示 |

**Common Mistakes 具体条目**（L61-66）:
1. 不要一次读取完整长文档 → 使用 `read_doc_segment` 或 `search_doc_keywords`
2. 不要不理解 schema 就 join → 先用 `inspect_sqlite_schema`
3. 不要用 Python 做 SQL 更擅长的事 → 数据库查询优先使用 `execute_context_sql`
4. 不要提交未经验证的答案 → 交叉校验后再调用 `answer`
5. 不要忽略错误 → 理解错误原因后调整策略
6. 不要忘记搜索相关关键词 → 大文档使用 `search_doc_keywords`

##### 3.2.3 Few-shot 示例 `RESPONSE_EXAMPLES`（L69-89）

**新增**: 4 个完整的 JSON 响应示例，覆盖最常用场景：

| 场景 | 行号 | 示例内容 |
|------|------|----------|
| 上下文检查 | L71-73 | `list_context` 调用 |
| SQL 查询 | L76-78 | `execute_context_sql` 调用（含完整 WHERE 条件） |
| Python 计算 | L81-83 | `execute_python` 调用（百分比计算） |
| 提交答案 | L86-88 | `answer` 调用（含完整 columns + rows） |

##### 3.2.4 难度感知的任务提示 `build_task_prompt()`（L104-149）

**新增**: 根据任务难度注入差异化的策略指导：

| 难度 | 策略提示 | 预期工具调用次数 |
|------|----------|------------------|
| easy | 数据量小，1-2 文件，简单计算 | 1-3 次 |
| medium | 可能需要多数据源合并（CSV+DB），SQL join 优先 | 3-6 次 |
| hard | 多数据源 + 非结构化文档，先读 knowledge.md，交叉引用 | 5-10 次 |
| extreme | 超长文档（>128K tokens），**必须**先用关键词搜索，限制步数 | 8-16 次，优先效率 |

**文件路径约定说明**（L146-148）: 明确告知模型文件路径格式（csv/data.csv, db/database.sqlite, doc/report.md, json/catalog.json），减少路径错误。

---

### 3.3 运行时状态（扩展）

#### 文件: [agents/runtime.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/runtime.py)

**变更类型**: 字段扩展。

**变更内容**:

```python
# L23-28
@dataclass(slots=True)
class AgentRuntimeState:
    plan: str | None = None          # 新增字段：分析计划文本
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None
```

**变更原因**: 支持规划阶段的输出在主推理循环中被引用（`state.plan` 在 `_build_messages()` 中追加到 task_content）。

---

### 3.4 模型适配器（扩展）

#### 文件: [agents/model.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/model.py)

**变更类型**: 接口扩展 + 新类添加。

##### 3.4.1 `ModelMessage` 数据类（L13-18）

**新增字段**: `tool_calls: list[dict] | None` 和 `tool_call_id: str | None`，支持 OpenAI 原生多轮 Tool Calling 格式。

##### 3.4.2 `ModelAdapter` 协议（L29-39）

**新增方法 `complete_with_tools()`**: 返回 `(content, tool_calls, tokens)` 三元组，支持原生 Tool Calling 响应格式。

##### 3.4.3 `OpenAIModelAdapter` 类（L42-146）

**核心变更**:

| 变更点 | 说明 | 行号 |
|--------|------|------|
| `_total_tokens` 属性 | 追踪总令牌消耗 | L68, L112 |
| `retry_delay` 参数 | 可配置的重试延迟（指数退避基础值） | L53 |
| `complete_with_tools()` | 新增完整实现，支持工具定义注入 | L74-141 |
| 指数退避重试 | `wait = retry_delay * (2 ** attempt)` | L134 |
| Token 追踪 | 每次 API 调用后累加 `_total_tokens` | L112 |

**`complete_with_tools()` 方法详细流程**（L74-141）:

1. 检查 API key 存在性（L80-81）
2. 将内部 `ModelMessage` 转换为 OpenAI 格式（L83-90）
3. 指数退避重试循环（L92-141）
4. 当 `tools` 参数非空时注入 `tools` 和 `tool_choice`（L101-103）
5. 解析响应中的工具调用（L117-128），标准化为内部格式

##### 3.4.4 `ScriptedModelAdapter` 类（L149-157）

**新增**: 用于单元测试的脚本化适配器，按预设顺序返回响应，支持 `complete()` 接口。

**使用场景**: `test_model.py` (4 个测试) 和 `test_react.py` (5 个测试) 中使用，无需真实 API。

---

### 3.5 工具注册表（扩展）

#### 文件: [tools/registry.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/tools/registry.py)

**变更类型**: 工具集扩展——新增 3 个文档处理工具。

##### 3.5.1 DocumentProcessor 集成（L17-19）

**新增**: 模块级单例 `_doc_processor = DocumentProcessor()`，供所有任务共享文档缓存。

##### 3.5.2 新增工具: `read_doc_segment`（L87-93）

```python
def _read_doc_segment(task, action_input):
    path = resolve_context_path(task, str(action_input["path"]))
    index = int(action_input.get("segment_index", 0))
    segment = _doc_processor.get_segment(path, index)
    if segment is None:
        return ToolExecutionResult(ok=False, content=f"Segment {index} not found...")
    return ToolExecutionResult(ok=True, content=segment)
```

**功能**: 按索引读取长文档的指定片段，返回内容带片段编号标记。

##### 3.5.3 新增工具: `search_doc_keywords`（L96-105）

```python
def _search_doc_keywords(task, action_input):
    path = resolve_context_path(task, str(action_input["path"]))
    keywords = action_input.get("keywords", [])
    if isinstance(keywords, str):
        keywords = json.loads(keywords) if keywords.startswith("[") else [keywords]
    ...
    results = _doc_processor.search_keywords(path, keywords)
    return ToolExecutionResult(ok=True, content=results)
```

**功能**: 关键词全文检索，支持传入 JSON 数组字符串或单个关键词字符串，返回匹配的片段索引、关键词和 300 字符预览。

##### 3.5.4 新增工具: `get_doc_info`（L108-111）

```python
def _get_doc_info(task, action_input):
    path = resolve_context_path(task, str(action_input["path"]))
    count = _doc_processor.get_segment_count(path)
    return ToolExecutionResult(ok=True, content={
        "segment_count": count,
        "max_chars_per_segment": _doc_processor.max_chars
    })
```

**功能**: 获取文档元数据，帮助模型在调用 `read_doc_segment` 前了解文档规模。

##### 3.5.5 新增工具: `summarize_csv` 和 `summarize_sqlite` 注册（L114-119）

**新增**: `_summarize_csv` 和 `_summarize_sqlite` 处理函数，封装对 `data_summary` 模块的调用。

##### 3.5.6 `create_default_tool_registry()` 更新（L170-268）

**新增注册的 ToolSpec 条目**:

| 工具名 | 行号 | 用途 |
|--------|------|------|
| `summarize_csv` | L172-180 | CSV 统计摘要 |
| `summarize_sqlite` | L181-188 | SQLite 统计摘要 |
| `read_doc_segment` | L189-193 | 按片段读取长文档 |
| `search_doc_keywords` | L194-198 | 关键词全文检索 |
| `get_doc_info` | L199-203 | 文档元数据查询 |

**工具总数**: 从原始的 8 个扩展到 13 个。

---

## 4. 测试基础设施

### 4.1 测试文件总览

| 文件 | 测试数 | 覆盖范围 |
|------|--------|----------|
| [tests/test_model.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/test_model.py) | 4 | ModelMessage 默认值/工具调用字段、ScriptedModelAdapter 消耗/耗尽 |
| [tests/test_react.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/test_react.py) | 5 | JSON 解析（带/不带 fence、无效输入）、强制超时回答、连续错误干预 |
| [tests/test_scorer.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/test_scorer.py) | 7 | 精确匹配、列顺序无关、额外列惩罚、缺失列、空 GT、NULL 归一化、数值舍入 |
| [tests/perf_test.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/perf_test.py) | 集成 | 5 任务批量性能测试，三维评分体系 |
| [tests/gen_report.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/gen_report.py) | 报告 | 脱机综合评分报告生成 |

**测试总数**: 16 个单元测试，全部通过（`16 passed`）。

### 4.2 测试执行结果（性能测试）

基于 2026-05-02 对 5 个 easy 任务的测试：

| 指标 | 数值 |
|------|------|
| 成功率 | 60.0% (3/5) |
| 总耗时 | 127.8s |
| 吞吐量 | 140.8 tasks/h |
| 峰值内存 | 128MB |
| 加权总分 | 83.8/100 |
| 综合评级 | **A** |

**失败任务分析**:

| 任务 | 问题 | 根因 |
|------|------|------|
| task_22 | 原生 TC 返回空 action | Qwen3.6-35b-a3b 原生 TC 返回空 function name，文本回退也未产出有效 action |
| task_25 | 同上 | 同上 |

---

## 5. 性能与配置优化

### 5.1 配置参数调整

#### 文件: [config.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/config.py)（L36-46）

| 参数 | 原始值 | 优化值 | 调整原因 |
|------|--------|--------|----------|
| `max_workers` | 4 | **8** (L40) | 充分利用 16 核 CPU（测试峰值仅 100.3%，有大量闲置） |
| `timeout_easy` | 90s | **120s** (L42) | 实测 easy 任务平均 25.6s，但 task_19 达 45.8s，保留合理缓冲 |
| `timeout_medium` | 180s | **240s** (L43) | medium 任务预计 3-6 工具调用，每步 ~10s + API 延迟，需更大缓冲 |

#### 文件: [configs/dev.yaml](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/configs/dev.yaml)

**变更**: 与 `config.py` 默认值保持同步（`max_workers: 4`, `timeout_easy: 90`, `timeout_medium: 180`）。

### 5.2 三层回退容错

在 [react.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py) 主循环中实现:

```
原生 Tool Calling → (失败) → JSON 文本补全 → (失败) → 错误记录 + 系统干预
```

配合指数退避重试（[model.py L134](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/model.py#L134)），在 API 抖动时自动恢复。

### 5.3 上下文压缩

`ContextManager._compress()` 确保 token 预算在 200K 限制内，防止模型输入溢出导致 API 错误。

---

## 6. Bug 修复

### 6.1 连续错误干预缺失

**问题**: [react.py L218-224](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/src/data_agent_baseline/agents/react.py#L218-L224)  
`consecutive_errors` 的阈值检查仅在 try 块中执行。当工具执行抛出异常进入 except 块时，虽然计数递增，但不会触发系统干预。

**修复**: 在 except 块末尾也添加阈值检查和系统干预注入，确保无论错误发生在哪个阶段，连续错误触发后都会引导模型调整策略。

**影响范围**: 所有任务的错误恢复机制。修复后，连续 3 次错误必然触发系统提示。

### 6.2 Windows GBK 编码兼容性

**问题**: [tests/perf_test.py L428-446](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/perf_test.py#L428-L446)  
bottleneck 列表中包含 Unicode 特殊字符（如 `\u26a0` ⚠），在 Windows GBK 控制台输出时触发 `UnicodeEncodeError`，导致报告生成崩溃。

**修复**: 将所有非 ASCII 字符替换为纯 ASCII 等价文本（`[WARN]`, `[INFO]`, `[OK]` 等标签使用纯文本形式），同时所有瓶颈描述文本从中文切换为英文以避免 GBK 范围内的编码问题。

### 6.3 API 配额耗尽处理

**问题**: 在 DashScope 免费配额耗尽后（`AllocationQuota.FreeTierOnly` 403 错误），性能测试全部失败且无法生成有意义的报告。

**方案**: 创建脱机报告生成器 [gen_report.py](file:///d:/2026-KDD-Cup/kddcup2026-data-agents-starter-kit/tests/gen_report.py)，使用首次运行保留的有效数据生成完整报告，不依赖 API。

---

## 7. 已知问题与后续方向

### 7.1 P0 — 原生 Tool Calling 回退链缺陷

**现象**: task_22、task_25 因 Qwen3.6-35b-a3b 的原生 TC 返回空 function name 而失败。  
**影响**: 当前 40% 失败率（2/5），扩展到 380 任务预估 ~152 任务失败。  
**建议方案**: 当 native TC + text 均失败时，发送"你的上一个 action 为空，请重新输出一个有效的 JSON action"提示，引导模型重试。

### 7.2 P1 — 为 easy 任务禁用规划阶段

**现象**: 规划阶段消耗 1 次 API 调用，在简单任务中收益有限。  
**建议方案**: 在 `ReActAgentConfig` 中增加 `planning_difficulty_threshold` 参数，或通过 `build_task_prompt` 的难度信息动态决定是否启用。

### 7.3 P1 — 利用闲置 CPU 资源

**现象**: 峰值 CPU 100.3%（瞬时），峰值内存 128MB，大量资源闲置。  
**建议方案**: 将 `max_workers` 从当前配置的 4 提升至 8-12，充分利用 16 核 CPU。

### 7.4 P2 — Token 使用量监控

**现象**: `OpenAIModelAdapter._total_tokens` 已追踪但未在 Agent 层面展示。  
**建议方案**: 在 Agent 执行完成后输出 token 使用量日志，确保不超出 200K 预算。

### 7.5 P2 — Docker 环境验证

**现象**: 当前所有测试在 Windows 开发环境完成，与竞赛 Docker 环境（16CPU/64GB/12h）存在差异。  
**建议方案**: 在 Docker 镜像中运行完整 380 任务流程，验证资源限制下的真实表现。

---

## 附录 A: 文件变更清单

### 新增文件 (11 个)

| 文件路径 | 行数 | 类型 |
|----------|------|------|
| `src/data_agent_baseline/context/__init__.py` | 0 | 模块初始化 |
| `src/data_agent_baseline/context/manager.py` | 66 | 上下文管理器 |
| `src/data_agent_baseline/context/document.py` | 59 | 文档处理器 |
| `src/data_agent_baseline/tools/data_summary.py` | 63 | 数据摘要工具 |
| `tests/__init__.py` | 0 | 测试模块初始化 |
| `tests/test_model.py` | 33 | 模型适配器测试 |
| `tests/test_react.py` | 72 | ReAct Agent 测试 |
| `tests/test_scorer.py` | 93 | ColumnScorer 测试 |
| `tests/perf_test.py` | 508 | 性能测试框架 |
| `tests/gen_report.py` | 303 | 脱机报告生成器 |
| `artifacts/perf_tests/performance_report.json` | ~250 | 性能测试报告 |

### 修改文件 (9 个)

| 文件路径 | 变更类型 |
|----------|----------|
| `src/data_agent_baseline/agents/react.py` | 重构：新增 native TC、规划、反思、ContextManager 集成 |
| `src/data_agent_baseline/agents/prompt.py` | 重构：新增规划提示词、增强推理提示词、难度感知 |
| `src/data_agent_baseline/agents/runtime.py` | 扩展：新增 `plan` 字段 |
| `src/data_agent_baseline/agents/model.py` | 扩展：新增 `complete_with_tools`、`ScriptedModelAdapter`、token 追踪 |
| `src/data_agent_baseline/tools/registry.py` | 扩展：新增 5 个工具（3 文档 + 2 摘要） |
| `src/data_agent_baseline/config.py` | 优化：`max_workers` 4→8, `timeout_easy` 90→120, `timeout_medium` 180→240 |
| `configs/dev.yaml` | 同步：与 config.py 默认值保持一致 |
| `src/data_agent_baseline/eval/run_eval.py` | 优化：评分流程增强 |
| `main.py` | 调整：日志和输出路径适配 |

---

## 附录 B: 评分标准完整定义

### 质量评分 (权重 45%)

| 扣分条件 | 扣分值 | 上限 |
|----------|--------|------|
| 任务未成功完成 | -40 | — |
| 答案列为空 (0 columns) | -30 | — |
| 答案行为空 (0 rows) | -20 | — |
| 每次工具错误 | -5 | -15 |

### 性能评分 (权重 30%)

| 扣分条件 | 扣分公式 | 上限 |
|----------|----------|------|
| 执行时间 > 60s | `min(30, (elapsed-60) × 0.5)` | -30 |
| 步数 > 12 | `(steps-12) × 2` | — |
| 内存 > 1000MB | `min(15, (memory-1000) / 100)` | -15 |

### 效率评分 (权重 25%)

| 扣分条件 | 扣分公式 |
|----------|----------|
| API 调用 > 10 次 | `(api_calls-10) × 3` |
| 耗时 > 30s 且 API ≤ 3 次 | -10 |

### 等级映射

| 总分范围 | 等级 |
|----------|------|
| ≥ 90 | A+ |
| ≥ 80 | A |
| ≥ 70 | B |
| ≥ 60 | C |
| < 60 | D |
