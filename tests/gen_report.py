from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

report_data = {
    "report_metadata": {
        "generated_at": datetime.now().isoformat(),
        "test_config": "configs/dev.yaml",
        "task_count": 5,
        "model": "qwen3.6-35b-a3b",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "summary": {
        "total_elapsed_seconds": 127.8,
        "success_rate_pct": 60.0,
        "throughput_tasks_per_hour": 140.8,
        "total_api_calls": 22,
        "total_errors": 0,
        "average_quality_score": 64.0,
        "average_performance_score": 100.0,
        "average_efficiency_score": 100.0,
        "weighted_total_score": 83.8,
        "grade": "A",
        "peak_cpu_pct": 100.3,
        "peak_memory_mb": 128,
        "bottlenecks": [
            "[WARN] Task_22/Task_25 因原生 Tool Calling 返回空 action 导致失败，需增强回退逻辑",
            "[INFO] 总耗时 127.8s / 5 任务 = 25.6s/task，扩展到 380 任务预估 ~2.7h，在 12h 时限内安全",
        ],
    },
    "tasks": [
        {
            "task_id": "task_11",
            "difficulty": "easy",
            "question": "For patients with severe degree of thrombosis, list their ID, sex and disease the patient is diagnosed with.",
            "elapsed_seconds": 33.32,
            "step_count": 7,
            "tool_calls": 6,
            "tool_errors": 0,
            "model_api_calls": 7,
            "succeeded": True,
            "failure_reason": None,
            "answer_columns": 3,
            "answer_rows": 3,
            "peak_cpu_pct": 100.3,
            "peak_memory_mb": 75,
            "avg_cpu_pct": 2.0,
            "avg_memory_mb": 74,
            "disk_read_mb": 0.0,
            "disk_write_mb": 0.0,
            "quality_score": 100.0,
            "quality_issues": ["No issues detected"],
            "performance_score": 100.0,
            "performance_issues": ["Performance within acceptable range"],
            "efficiency_score": 100.0,
            "efficiency_issues": ["Resource usage efficient"],
        },
        {
            "task_id": "task_19",
            "difficulty": "easy",
            "question": "List the full name of the Student_Club members that grew up in Illinois state.",
            "elapsed_seconds": 45.82,
            "step_count": 9,
            "tool_calls": 7,
            "tool_errors": 0,
            "model_api_calls": 8,
            "succeeded": True,
            "failure_reason": None,
            "answer_columns": 1,
            "answer_rows": 3,
            "peak_cpu_pct": 66.5,
            "peak_memory_mb": 128,
            "avg_cpu_pct": 0.6,
            "avg_memory_mb": 112,
            "disk_read_mb": 0.0,
            "disk_write_mb": 0.0,
            "quality_score": 100.0,
            "quality_issues": ["No issues detected"],
            "performance_score": 100.0,
            "performance_issues": ["Performance within acceptable range"],
            "efficiency_score": 100.0,
            "efficiency_issues": ["Resource usage efficient"],
        },
        {
            "task_id": "task_22",
            "difficulty": "easy",
            "question": "State the date Connor Hilton paid his/her dues.",
            "elapsed_seconds": 11.81,
            "step_count": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "model_api_calls": 2,
            "succeeded": False,
            "failure_reason": "action must be a non-empty string (Native TC parsing error)",
            "answer_columns": 0,
            "answer_rows": 0,
            "peak_cpu_pct": 0.0,
            "peak_memory_mb": 128,
            "avg_cpu_pct": 0.0,
            "avg_memory_mb": 128,
            "disk_read_mb": 0.0,
            "disk_write_mb": 0.0,
            "quality_score": 10.0,
            "quality_issues": ["Task did not complete successfully", "Empty answer (0 columns)", "Empty answer (0 rows)"],
            "performance_score": 100.0,
            "performance_issues": ["Performance within acceptable range"],
            "efficiency_score": 100.0,
            "efficiency_issues": ["Resource usage efficient"],
        },
        {
            "task_id": "task_24",
            "difficulty": "easy",
            "question": "How many members attended the \"Women's Soccer\" event?",
            "elapsed_seconds": 24.93,
            "step_count": 7,
            "tool_calls": 6,
            "tool_errors": 0,
            "model_api_calls": 7,
            "succeeded": True,
            "failure_reason": None,
            "answer_columns": 2,
            "answer_rows": 1,
            "peak_cpu_pct": 0.0,
            "peak_memory_mb": 128,
            "avg_cpu_pct": 0.0,
            "avg_memory_mb": 128,
            "disk_read_mb": 0.0,
            "disk_write_mb": 0.0,
            "quality_score": 100.0,
            "quality_issues": ["No issues detected"],
            "performance_score": 100.0,
            "performance_issues": ["Performance within acceptable range"],
            "efficiency_score": 100.0,
            "efficiency_issues": ["Resource usage efficient"],
        },
        {
            "task_id": "task_25",
            "difficulty": "easy",
            "question": "Which event has the lowest cost?",
            "elapsed_seconds": 11.94,
            "step_count": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "model_api_calls": 2,
            "succeeded": False,
            "failure_reason": "action must be a non-empty string (Native TC parsing error)",
            "answer_columns": 0,
            "answer_rows": 0,
            "peak_cpu_pct": 0.0,
            "peak_memory_mb": 128,
            "avg_cpu_pct": 0.0,
            "avg_memory_mb": 128,
            "disk_read_mb": 0.0,
            "disk_write_mb": 0.0,
            "quality_score": 10.0,
            "quality_issues": ["Task did not complete successfully", "Empty answer (0 columns)", "Empty answer (0 rows)"],
            "performance_score": 100.0,
            "performance_issues": ["Performance within acceptable range"],
            "efficiency_score": 100.0,
            "efficiency_issues": ["Resource usage efficient"],
        },
    ],
}

output_dir = Path(r"D:\2026-KDD-Cup\kddcup2026-data-agents-starter-kit\artifacts\perf_tests")
output_dir.mkdir(parents=True, exist_ok=True)

report_path = output_dir / "performance_report.json"
report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n")

print("=" * 80)
print("              DATA AGENT 综合性能评分报告")
print("=" * 80)
print(f"  报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  测试任务数: 5  |  模型: qwen3.6-35b-a3b")
print(f"  成功率: 60.0% (3/5)")
print("=" * 80)

print("\n## 一、任务执行概览\n")
print(f"  {'Task':<10} {'难度':<8} {'耗时':>8} {'步数':>5} {'工具调用':>8} {'错误':>5} {'状态':<6} {'答案':>6} {'质量':>5} {'性能':>5} {'效率':>5}")
print(f"  {'-'*85}")

metrics = report_data["tasks"]
total_elapsed = 0
total_steps = 0
total_errors = 0
total_api = 0
all_quality = []
all_perf = []
all_eff = []

for m in metrics:
    status = "OK" if m["succeeded"] else "FAIL"
    ans = f"{m['answer_columns']}Cx{m['answer_rows']}R"
    q = m["quality_score"]
    p = m["performance_score"]
    e = m["efficiency_score"]
    print(f"  {m['task_id']:<10} {m['difficulty']:<8} {m['elapsed_seconds']:>7.1f}s {m['step_count']:>5} {m['tool_calls']:>8} {m['tool_errors']:>5} {status:<6} {ans:>6} {q:>5.0f} {p:>5.0f} {e:>5.0f}")
    total_elapsed += m["elapsed_seconds"]
    total_steps += m["step_count"]
    total_errors += m["tool_errors"]
    total_api += m["model_api_calls"]
    all_quality.append(q)
    all_perf.append(p)
    all_eff.append(e)

print(f"\n## 二、逐任务分析\n")

for m in metrics:
    status = "SUCCESS" if m["succeeded"] else "FAILED"
    print(f"\n### {m['task_id']} [{status}]")
    print(f"  Question: {m['question'][:80]}...")
    print(f"  Duration: {m['elapsed_seconds']:.1f}s | Steps: {m['step_count']} | Tool calls: {m['tool_calls']} | API calls: {m['model_api_calls']}")
    print(f"  Answer: {m['answer_columns']} columns x {m['answer_rows']} rows")
    if not m["succeeded"]:
        print(f"  Failure: {m['failure_reason']}")
        print(f"  Root cause: Native Tool Calling returned tool_calls with empty function name. Qwen3.5 model occasionally returns malformed tool call structures. The fallback text completion also failed to produce a valid JSON action.")
    else:
        print(f"  Quality issues: {', '.join(m['quality_issues'])}")
    print(f"  Scores: Q={m['quality_score']:.0f} P={m['performance_score']:.0f} E={m['efficiency_score']:.0f}")

avg_q = sum(all_quality) / len(all_quality)
avg_p = sum(all_perf) / len(all_perf)
avg_e = sum(all_eff) / len(all_eff)
success_rate = sum(1 for m in metrics if m["succeeded"]) / len(metrics) * 100
wt = avg_q * 0.45 + avg_p * 0.30 + avg_e * 0.25
throughput = len(metrics) / (total_elapsed / 3600) if total_elapsed > 0 else 0
grade = "A" if wt >= 80 else "B" if wt >= 70 else "C" if wt >= 60 else "D"

print(f"\n## 三、汇总指标\n")
print(f"  | 指标                 | 数值                          |")
print(f"  |----------------------|-------------------------------|")
print(f"  | 任务总数             |    5                          |")
print(f"  | 成功 / 失败          |  3 / 2                        |")
print(f"  | 成功率               |  {success_rate:.1f}%                        |")
print(f"  | 总耗时               |  {total_elapsed:.1f}s                      |")
print(f"  | 平均耗时             |  {total_elapsed/len(metrics):.1f}s/任务                  |")
print(f"  | 吞吐量               |  {throughput:.1f} tasks/h                 |")
print(f"  | 总步数               |  {total_steps}                           |")
print(f"  | 总工具调用           |  {sum(m['tool_calls'] for m in metrics)}                           |")
print(f"  | 总API调用            |  {total_api}                           |")
print(f"  | 峰值CPU              |  {max(m['peak_cpu_pct'] for m in metrics):.1f}%                        |")
print(f"  | 峰值内存             |  {max(m['peak_memory_mb'] for m in metrics):.0f}MB                     |")

print(f"\n## 四、加权综合评分\n")
print(f"  | 维度         | 权重   | 得分    | 加权得分    | 评价                            |")
print(f"  |--------------|--------|---------|-------------|---------------------------------|")
print(f"  | 任务完成质量 | 45%    |  {avg_q:>5.1f}   | {avg_q*0.45:>6.1f}       | 3/5 任务成功，2 任务因原生TC失败   |")
print(f"  | 执行性能     | 30%    | {avg_p:>5.1f}   | {avg_p*0.30:>6.1f}       | 所有任务响应时间正常              |")
print(f"  | 资源效率     | 25%    | {avg_e:>5.1f}   | {avg_e*0.25:>6.1f}       | CPU/内存使用极低                  |")
print(f"  | **加权总分** | **100%** |         | **{wt:>5.1f}**     | **等级: {grade}**                         |")

print(f"\n## 五、性能瓶颈分析\n")

bottlenecks = [
    {
        "severity": "HIGH",
        "issue": "原生 Tool Calling 回退链缺陷",
        "detail": "task_22/task_25 均因 Qwen3.6-35b-a3b 的 native Tool Calling 返回空 action 名称而失败。虽然实现了 JSON 文本回退，但回退后的文本补全也未能产出有效的 action。",
        "impact": "影响 40% 的任务（2/5），如果扩展到全量 380 任务可能导致 ~152 任务失败",
        "recommendation": "增加更鲁棒的回退策略：当 native TC+text 均失败时，发送空白 action 错误提示引导模型重试"
    },
    {
        "severity": "MEDIUM",
        "issue": "规划阶段与主循环竞争 token 预算",
        "detail": "规划阶段消耗 1 次 API 调用，在某些简单任务中可能不必要。但它成功降级（plan failure 不阻止执行）。",
        "impact": "每任务增加 1 次 API 调用，轻微增加延迟 (~0.5-2s)",
        "recommendation": "为 easy 任务禁用规划阶段，或在首次工具调用后动态决定是否规划"
    },
    {
        "severity": "LOW",
        "issue": "任务 19 耗时最长 (45.8s)",
        "detail": "涉及跨文件数据合并（CSV+JSON+knowledge.md），Step 数达 9 步，工具调用 7 次。",
        "impact": "单任务耗时在可接受范围内",
        "recommendation": "可考虑为 medium+ 任务增加并行工具调用能力"
    },
    {
        "severity": "INFO",
        "issue": "内存/CPU 资源充裕",
        "detail": "峰值内存仅 128MB，峰值 CPU 100.3%（瞬间），远低于 64GB/16核限制。",
        "impact": "有大量资源可用于并行执行更多任务",
        "recommendation": "可将 max_workers 从 4 提升至 8-12，充分利用 16 核 CPU"
    },
]

for i, b in enumerate(bottlenecks):
    print(f"\n  [{b['severity']}] {b['issue']}")
    print(f"    Detail: {b['detail']}")
    print(f"    Impact: {b['impact']}")
    print(f"    Fix:    {b['recommendation']}")

print(f"\n## 六、建议与后续优化方向\n")
print(f"  1. [P0] 修复原生 Tool Calling 回退链 — 添加空白 action 重试逻辑")
print(f"  2. [P1] 为 easy 任务禁用规划阶段，节省 ~20% API 调用")
print(f"  3. [P1] 增加并发度 max_workers: 4 -> 8，利用闲置 CPU")
print(f"  4. [P2] 为 medium+ 任务探索并行工具调用（branch-and-merge）")
print(f"  5. [P2] 添加 token 使用量监控，确保不超出 200K 预算")
print(f"  6. [P2] 在 Docker 环境（16CPU/64GB/12h）中验证完整 380 任务流程")
print(f"\n{'='*80}")
print(f"  详细报告已保存至: {report_path}")
print(f"{'='*80}")
