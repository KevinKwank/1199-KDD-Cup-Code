from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.filesystem import resolve_context_path


def summarize_csv(task: PublicTask, action_input: dict[str, Any]) -> dict[str, Any]:
    path = resolve_context_path(task, str(action_input["path"]))

    import pandas as pd
    df = pd.read_csv(path)

    result: dict[str, Any] = {
        "path": str(action_input["path"]),
        "shape": [df.shape[0], df.shape[1]],
        "columns": [],
    }

    for col in df.columns:
        col_info: dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "unique_count": int(df[col].nunique()),
        }
        if df[col].nunique() <= 15:
            col_info["unique_values"] = df[col].dropna().unique().tolist()
        result["columns"].append(col_info)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        stats = df[numeric_cols].describe().to_dict()
        result["numeric_stats"] = {
            col: {k: round(v, 2) if isinstance(v, float) else v for k, v in vals.items()}
            for col, vals in stats.items()
        }

    return result


def summarize_sqlite(task: PublicTask, action_input: dict[str, Any]) -> dict[str, Any]:
    path = resolve_context_path(task, str(action_input["path"]))

    import sqlite3
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cursor.fetchall()]

    result: dict[str, Any] = {"path": str(action_input["path"]), "tables": []}
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        count = cursor.fetchone()[0]
        cursor.execute(f'PRAGMA table_info("{table}")')
        cols = [{"name": c[1], "type": c[2]} for c in cursor.fetchall()]
        result["tables"].append({"name": table, "row_count": count, "columns": cols})

    conn.close()
    return result
