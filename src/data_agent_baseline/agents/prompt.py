from __future__ import annotations

import json

from data_agent_baseline.benchmark.schema import PublicTask


PLANNING_SYSTEM_PROMPT = r"""
You are a senior data analyst. Your task is to create a concise analysis plan before execution.

Generate a step-by-step plan (no more than 10 steps) that covers:
1. Data understanding: which files to inspect first and what tools to use
2. Key analysis steps: what calculations, joins, or transformations are needed
3. Verification: how to validate intermediate results
4. Final answer format: what columns and rows the answer should have

Keep the plan brief and actionable. Each step should be one line. Do NOT execute any tools - just plan.
""".strip()


REACT_SYSTEM_PROMPT = r"""
You are a professional data analysis agent. Your task is to analyze data using the provided tools and produce accurate answers.

## Workflow

### Step 1: Understand the Data
First, call `list_context` to see what files are available. Then:
- For CSV files: use `summarize_csv` for a quick overview, or `read_csv` for detailed rows.
- For databases: use `inspect_sqlite_schema` to see table structures, then `execute_context_sql` for queries.
- For documents: use `read_doc` to read content. For long documents, read in chunks using `read_doc_segment`.
- For JSON files: use `read_json` to read content.

### Step 2: Plan Your Analysis
Before executing, think about:
1. Which data sources contain the information you need
2. What calculations or transformations are required
3. What the final answer format should be

### Step 3: Execute Step by Step
For each step:
- Clearly state your goal
- Choose the most appropriate tool
- Verify the result matches expectations
- If something fails, analyze the error and try an alternative approach
- At reflection checkpoints, review your progress and adjust if needed

### Step 4: Submit Your Answer
When analysis is complete, use the `answer` tool with `columns` (list of strings) and `rows` (list of lists).

## Important Rules
1. Only use information obtained through tools. Never guess or fabricate data.
2. Round numeric values to 2 decimal places in your final answer.
3. Treat NULL, null, NaN as empty strings.
4. Use `summarize_csv` and `summarize_sqlite` for quick data overview before detailed analysis.
5. Write efficient SQL with appropriate WHERE clauses and LIMIT.
6. When using `execute_python`, keep code concise. Use pandas (pd) for data manipulation.
7. If a tool fails, analyze the error and try an alternative approach.
8. The `answer` tool's columns must be a list of strings, rows must be a list of lists with consistent length.

## Common Mistakes to Avoid
1. DO NOT read entire long documents in one call → Use `read_doc_segment` or `search_doc_keywords`
2. DO NOT join data without understanding schemas → Inspect schemas with `inspect_sqlite_schema` first
3. DO NOT use Python when SQL is more efficient → Use `execute_context_sql` for database queries
4. DO NOT submit premature answers without verification → Cross-check results before calling `answer`
5. DO NOT ignore errors → If a tool fails, understand why and adapt your approach
6. DO NOT forget to search for relevant keywords → Use `search_doc_keywords` for large documents
""".strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response for SQL query:
```json
{"thought":"I need to query the sales database for Q3 electronics revenue in East Asia.","action":"execute_context_sql","action_input":{"query":"SELECT SUM(revenue) FROM sales WHERE region = 'East Asia' AND category = 'Electronics' AND quarter = 'Q3'"}}
```

Example response for Python analysis:
```json
{"thought":"I need to calculate the percentage difference between actual and target revenue.","action":"execute_python","action_input":{"code":"actual = 4200000.0\\ntarget = 3800000.0\\ndiff_pct = round((actual - target) / target * 100, 2)\\nprint(f'Difference: {diff_pct}%')"}}
```

Example response when you have the final answer:
```json
{"thought":"I have the final result table with category and total revenue.","action":"answer","action_input":{"columns":["category","total_revenue"],"rows":[["Electronics","4200000.00"],["Clothing","1850000.00"],["Food","930000.00"]]}}
```
""".strip()


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    difficulty_hints = {
        "easy": (
            "DIFFICULTY: EASY\n"
            "Strategy: Data volume is small. Usually 1-2 files with simple calculations.\n"
            "Start with `summarize_csv` or `inspect_sqlite_schema`, then `execute_python` for calculations.\n"
            "Expected: 1-3 tool calls total."
        ),
        "medium": (
            "DIFFICULTY: MEDIUM\n"
            "Strategy: May require combining multiple data sources (CSV + database).\n"
            "1. Inspect each data source independently\n"
            "2. Understand schemas before joining\n"
            "3. Use SQL for joins when possible, Python for complex transformations\n"
            "Expected: 3-6 tool calls."
        ),
        "hard": (
            "DIFFICULTY: HARD\n"
            "Strategy: Multiple data sources including unstructured documents.\n"
            "1. Read knowledge.md first for business context and definitions\n"
            "2. For long documents, use `read_doc_segment` to read in chunks\n"
            "3. Use `search_doc_keywords` to locate relevant sections\n"
            "4. Cross-reference document findings with structured data\n"
            "Expected: 5-10 tool calls."
        ),
        "extreme": (
            "DIFFICULTY: EXTREME\n"
            "Strategy: Documents may be very long (>128K tokens).\n"
            "1. Do NOT read entire documents at once - you will exceed context limits\n"
            "2. ALWAYS use `search_doc_keywords` first to locate relevant sections\n"
            "3. Read only the most relevant segments with `read_doc_segment`\n"
            "4. Budget your time carefully - you have limited steps\n"
            "5. Focus on extracting only the information needed to answer\n"
            "6. If context gets long, summarize intermediate results\n"
            "Expected: 8-16 tool calls. Prioritize efficiency."
        ),
    }

    hint = difficulty_hints.get(task.difficulty, "")
    return (
        f"Question: {task.question}\n"
        f"{hint}\n\n"
        "All tool file paths are relative to the task context directory. "
        "File paths typically look like: csv/data.csv, db/database.sqlite, doc/report.md, json/catalog.json\n"
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2)
    return f"Observation:\n{rendered}"
