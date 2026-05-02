from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.runtime import AgentRunResult
from data_agent_baseline.benchmark.schema import PublicTask, TaskRecord, TaskAssets
from data_agent_baseline.config import load_app_config
from data_agent_baseline.tools.registry import create_default_tool_registry


@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_read_mb: float
    disk_write_mb: float


@dataclass
class TaskPerfMetrics:
    task_id: str
    difficulty: str
    question: str
    started_at: str
    finished_at: str
    e2e_elapsed_seconds: float
    step_count: int
    succeeded: bool
    failure_reason: str | None
    answer_columns: int
    answer_rows: int
    model_api_calls: int
    tool_calls_made: int
    tool_errors: int
    plan_generated: bool
    peak_cpu_percent: float
    peak_memory_mb: float
    avg_cpu_percent: float
    avg_memory_mb: float
    disk_read_total_mb: float
    disk_write_total_mb: float
    snapshots: list[ResourceSnapshot] = field(default_factory=list)


class ResourceMonitor:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._snapshots: list[ResourceSnapshot] = []
        self._thread: threading.Thread | None = None
        self._process = psutil.Process()
        self._start_io = self._process.io_counters()
        self._start_disk = psutil.disk_io_counters()

    def start(self) -> None:
        self._stop.clear()
        self._snapshots = []
        self._start_io = self._process.io_counters()
        self._start_disk = psutil.disk_io_counters()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        while not self._stop.is_set():
            try:
                cpu = self._process.cpu_percent(interval=0.1)
                mem_info = self._process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                mem_percent = self._process.memory_percent()
                io = self._process.io_counters()
                disk = psutil.disk_io_counters()

                disk_read_mb = (io.read_bytes - self._start_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (io.write_bytes - self._start_io.write_bytes) / (1024 * 1024)

                self._snapshots.append(ResourceSnapshot(
                    timestamp=time.perf_counter(),
                    cpu_percent=cpu,
                    memory_mb=mem_mb,
                    memory_percent=mem_percent,
                    disk_read_mb=disk_read_mb,
                    disk_write_mb=disk_write_mb,
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop.wait(self.interval)

    @property
    def snapshots(self) -> list[ResourceSnapshot]:
        return list(self._snapshots)

    @property
    def peak_cpu(self) -> float:
        if not self._snapshots:
            return 0.0
        return max(s.cpu_percent for s in self._snapshots)

    @property
    def peak_memory_mb(self) -> float:
        if not self._snapshots:
            return 0.0
        return max(s.memory_mb for s in self._snapshots)

    @property
    def avg_cpu(self) -> float:
        if not self._snapshots:
            return 0.0
        return sum(s.cpu_percent for s in self._snapshots) / len(self._snapshots)

    @property
    def avg_memory_mb(self) -> float:
        if not self._snapshots:
            return 0.0
        return sum(s.memory_mb for s in self._snapshots) / len(self._snapshots)

    @property
    def disk_read_total_mb(self) -> float:
        if not self._snapshots:
            return 0.0
        return self._snapshots[-1].disk_read_mb

    @property
    def disk_write_total_mb(self) -> float:
        if not self._snapshots:
            return 0.0
        return self._snapshots[-1].disk_write_mb


def load_task(task_dir: Path) -> PublicTask:
    task_json_path = task_dir / "task.json"
    payload = json.loads(task_json_path.read_text())
    record = TaskRecord(
        task_id=payload["task_id"],
        difficulty=payload.get("difficulty", "unknown"),
        question=payload["question"],
    )
    assets = TaskAssets(task_dir=task_dir, context_dir=task_dir / "context")
    return PublicTask(record=record, assets=assets)


def run_performance_test(task_dirs: list[Path], config_path: Path) -> list[TaskPerfMetrics]:
    config = load_app_config(config_path)
    model = OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
        max_tokens=config.agent.max_tokens,
        timeout=config.agent.timeout,
        max_retries=config.agent.max_retries,
    )
    tools = create_default_tool_registry()
    agent_config = ReActAgentConfig(max_steps=config.agent.max_steps)

    results: list[TaskPerfMetrics] = []

    for task_dir in task_dirs:
        task = load_task(task_dir)
        print(f"\n{'='*70}")
        print(f"Running: {task.task_id} | {task.difficulty} | {task.question}")
        print(f"{'='*70}")

        monitor = ResourceMonitor(interval=0.3)
        agent = ReActAgent(model=model, tools=tools, config=agent_config)

        monitor.start()
        started_at = datetime.now()
        t0 = time.perf_counter()

        try:
            run_result: AgentRunResult = agent.run(task)
        except Exception as exc:
            run_result = AgentRunResult(
                task_id=task.task_id,
                answer=None,
                steps=[],
                failure_reason=str(exc),
            )

        elapsed = time.perf_counter() - t0
        finished_at = datetime.now()
        monitor.stop()

        steps = run_result.steps
        tool_steps = [s for s in steps if s.action not in ("__error__", "__system_hint__", "__reflect__")]
        error_steps = [s for s in steps if s.action == "__error__"]
        has_plan = bool(run_result.steps)

        answer_cols = 0
        answer_rows = 0
        if run_result.answer is not None:
            answer_cols = len(run_result.answer.columns)
            answer_rows = len(run_result.answer.rows)

        api_calls = len(tool_steps) + (1 if has_plan else 0)

        metrics = TaskPerfMetrics(
            task_id=task.task_id,
            difficulty=task.difficulty,
            question=task.question,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            e2e_elapsed_seconds=round(elapsed, 3),
            step_count=len(steps),
            succeeded=run_result.succeeded,
            failure_reason=run_result.failure_reason,
            answer_columns=answer_cols,
            answer_rows=answer_rows,
            model_api_calls=api_calls,
            tool_calls_made=len(tool_steps),
            tool_errors=len(error_steps),
            plan_generated=has_plan,
            peak_cpu_percent=round(monitor.peak_cpu, 2),
            peak_memory_mb=round(monitor.peak_memory_mb, 2),
            avg_cpu_percent=round(monitor.avg_cpu, 2),
            avg_memory_mb=round(monitor.avg_memory_mb, 2),
            disk_read_total_mb=round(monitor.disk_read_total_mb, 2),
            disk_write_total_mb=round(monitor.disk_write_total_mb, 2),
            snapshots=monitor.snapshots,
        )
        results.append(metrics)

        status = "OK" if run_result.succeeded else f"FAIL: {run_result.failure_reason}"
        print(f"  Time: {elapsed:.2f}s | Steps: {len(steps)} | Tool calls: {len(tool_steps)} | Errors: {len(error_steps)}")
        print(f"  CPU avg/peak: {monitor.avg_cpu:.1f}% / {monitor.peak_cpu:.1f}% | RAM avg/peak: {monitor.avg_memory_mb:.0f}MB / {monitor.peak_memory_mb:.0f}MB")
        print(f"  Answer: {answer_cols} cols x {answer_rows} rows | Status: {status}")

    return results


def score_task_quality(metrics: TaskPerfMetrics) -> dict[str, Any]:
    quality_score = 100.0
    issues = []

    if not metrics.succeeded:
        quality_score -= 40.0
        issues.append("Task did not complete successfully")
    if metrics.answer_columns == 0:
        quality_score -= 30.0
        issues.append("Empty answer (0 columns)")
    if metrics.answer_rows == 0:
        quality_score -= 20.0
        issues.append("Empty answer (0 rows)")
    if metrics.tool_errors > 0:
        penalty = min(15.0, metrics.tool_errors * 5.0)
        quality_score -= penalty
        issues.append(f"{metrics.tool_errors} tool errors (-{penalty})")

    return {
        "score": round(max(0.0, quality_score), 1),
        "issues": issues if issues else ["No issues detected"],
    }


def score_performance(metrics: TaskPerfMetrics) -> dict[str, Any]:
    perf_score = 100.0
    issues = []

    if metrics.e2e_elapsed_seconds > 60:
        penalty = min(30.0, (metrics.e2e_elapsed_seconds - 60) * 0.5)
        perf_score -= penalty
        issues.append(f"Slow execution: {metrics.e2e_elapsed_seconds:.1f}s (-{penalty:.1f})")
    if metrics.step_count > 12:
        perf_score -= (metrics.step_count - 12) * 2.0
        issues.append(f"High step count: {metrics.step_count}")
    if metrics.peak_memory_mb > 1000:
        penalty = min(15.0, (metrics.peak_memory_mb - 1000) / 100)
        perf_score -= penalty
        issues.append(f"High memory: {metrics.peak_memory_mb:.0f}MB (-{penalty:.1f})")

    return {
        "score": round(max(0.0, perf_score), 1),
        "issues": issues if issues else ["Performance within acceptable range"],
    }


def score_resource_efficiency(metrics: TaskPerfMetrics) -> dict[str, Any]:
    eff_score = 100.0
    issues = []

    if metrics.model_api_calls > 10:
        eff_score -= (metrics.model_api_calls - 10) * 3.0
        issues.append(f"High API calls: {metrics.model_api_calls}")
    if metrics.e2e_elapsed_seconds > 30 and metrics.model_api_calls <= 3:
        eff_score -= 10.0
        issues.append(f"Slow despite few API calls ({metrics.model_api_calls})")

    return {
        "score": round(max(0.0, eff_score), 1),
        "issues": issues if issues else ["Resource usage efficient"],
    }


def generate_report(results: list[TaskPerfMetrics], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n\n")
    print("=" * 80)
    print("              DATA AGENT 综合性能评分报告")
    print("=" * 80)
    print(f"  报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  测试任务数: {len(results)}")
    print(f"  测试配置: configs/dev.yaml")
    print("=" * 80)

    print("\n## 一、任务执行概览\n")
    print(f"  {'Task':<10} {'难度':<8} {'耗时':>8} {'步数':>5} {'工具调用':>8} {'错误':>5} {'状态':<6} {'质量分':>7} {'性能分':>7} {'效率分':>7}")
    print(f"  {'-'*80}")

    total_elapsed = 0.0
    total_steps = 0
    total_errors = 0
    total_api_calls = 0
    all_peak_cpu = []
    all_peak_mem = []
    all_quality = []
    all_perf = []
    all_eff = []

    task_reports = []

    for m in results:
        quality = score_task_quality(m)
        perf = score_performance(m)
        eff = score_resource_efficiency(m)

        status = "OK" if m.succeeded else "FAIL"
        print(f"  {m.task_id:<10} {m.difficulty:<8} {m.e2e_elapsed_seconds:>7.1f}s {m.step_count:>5} {m.tool_calls_made:>8} {m.tool_errors:>5} {status:<6} {quality['score']:>6.1f} {perf['score']:>6.1f} {eff['score']:>6.1f}")

        all_quality.append(quality["score"])
        all_perf.append(perf["score"])
        all_eff.append(eff["score"])
        all_peak_cpu.append(m.peak_cpu_percent)
        all_peak_mem.append(m.peak_memory_mb)
        total_elapsed += m.e2e_elapsed_seconds
        total_steps += m.step_count
        total_errors += m.tool_errors
        total_api_calls += m.model_api_calls

        task_reports.append({
            "task_id": m.task_id,
            "difficulty": m.difficulty,
            "question": m.question,
            "elapsed_seconds": m.e2e_elapsed_seconds,
            "step_count": m.step_count,
            "tool_calls": m.tool_calls_made,
            "tool_errors": m.tool_errors,
            "model_api_calls": m.model_api_calls,
            "succeeded": m.succeeded,
            "failure_reason": m.failure_reason,
            "answer_columns": m.answer_columns,
            "answer_rows": m.answer_rows,
            "peak_cpu_pct": m.peak_cpu_percent,
            "peak_memory_mb": m.peak_memory_mb,
            "avg_cpu_pct": m.avg_cpu_percent,
            "avg_memory_mb": m.avg_memory_mb,
            "disk_read_mb": m.disk_read_total_mb,
            "disk_write_mb": m.disk_write_total_mb,
            "quality_score": quality["score"],
            "quality_issues": quality["issues"],
            "performance_score": perf["score"],
            "performance_issues": perf["issues"],
            "efficiency_score": eff["score"],
            "efficiency_issues": eff["issues"],
        })

    avg_quality = sum(all_quality) / len(all_quality) if all_quality else 0
    avg_perf = sum(all_perf) / len(all_perf) if all_perf else 0
    avg_eff = sum(all_eff) / len(all_eff) if all_eff else 0
    success_rate = sum(1 for m in results if m.succeeded) / len(results) * 100 if results else 0

    weighted_total = avg_quality * 0.45 + avg_perf * 0.30 + avg_eff * 0.25

    throughput = len(results) / (total_elapsed / 3600) if total_elapsed > 0 else 0

    print(f"\n## 二、汇总指标\n")
    print(f"  | 指标                 | 数值                          |")
    print(f"  |----------------------|-------------------------------|")
    print(f"  | 任务总数             | {len(results):>4}                          |")
    print(f"  | 成功率               | {success_rate:>5.1f}%                        |")
    print(f"  | 总耗时               | {total_elapsed:>7.1f}s                      |")
    print(f"  | 平均耗时             | {total_elapsed/len(results):>7.1f}s/任务                  |")
    print(f"  | 吞吐量               | {throughput:>6.1f} tasks/h                 |")
    print(f"  | 总步数               | {total_steps:>4}                           |")
    print(f"  | 总错误数             | {total_errors:>4}                           |")
    print(f"  | 总API调用            | {total_api_calls:>4}                           |")
    print(f"  | 峰值CPU              | {max(all_peak_cpu):>6.1f}%                        |")
    print(f"  | 峰值内存             | {max(all_peak_mem):>7.0f}MB                     |")
    print(f"  | 平均质量分           | {avg_quality:>7.1f}                         |")
    print(f"  | 平均性能分           | {avg_perf:>7.1f}                         |")
    print(f"  | 平均效率分           | {avg_eff:>7.1f}                         |")

    print(f"\n## 三、加权综合评分\n")
    print(f"  | 维度         | 权重   | 得分    | 加权得分    |")
    print(f"  |--------------|--------|---------|-------------|")
    print(f"  | 任务完成质量 | 45%    | {avg_quality:>5.1f}   | {avg_quality*0.45:>6.1f}       |")
    print(f"  | 执行性能     | 30%    | {avg_perf:>5.1f}   | {avg_perf*0.30:>6.1f}       |")
    print(f"  | 资源效率     | 25%    | {avg_eff:>5.1f}   | {avg_eff*0.25:>6.1f}       |")
    print(f"  | **加权总分** | **100%** |         | **{weighted_total:>5.1f}**     |")

    grade = "A+" if weighted_total >= 90 else "A" if weighted_total >= 80 else "B" if weighted_total >= 70 else "C" if weighted_total >= 60 else "D"
    print(f"\n  [RESULT] 综合评级: **{grade}**  (总分: {weighted_total:.1f}/100)")

    print(f"\n## 四、性能瓶颈分析\n")
    bottlenecks = []

    if max(all_peak_mem) > 800:
        bottlenecks.append("[WARN] Memory peak high (" + f"{max(all_peak_mem):.0f}" + "MB), risk of OOM Kill")
    if total_elapsed > 300:
        bottlenecks.append("[WARN] Total time high (" + f"{total_elapsed:.1f}" + "s), 380 tasks may exceed 12h limit")
    if total_errors > 5:
        bottlenecks.append("[WARN] High error count (" + str(total_errors) + "), strengthen error recovery")
    if avg_quality < 80:
        bottlenecks.append("[WARN] Low quality score (" + f"{avg_quality:.1f}" + "), prompts or tool chain need optimization")
    if avg_perf < 80:
        bottlenecks.append("[WARN] Low performance score (" + f"{avg_perf:.1f}" + "), execution efficiency needs improvement")
    if throughput < 3:
        bottlenecks.append("[WARN] Low throughput (" + f"{throughput:.1f}" + " tasks/h), increase concurrency")

    if not bottlenecks:
        print("  [OK] No obvious performance bottlenecks detected, system running well.")
    else:
        for b in bottlenecks:
            print(f"  {b}")

    print(f"\n## 五、建议与后续优化方向\n")
    print(f"  1. Easy 任务预期应在 20-30s 内完成，若超时需检查 API 延迟")
    print(f"  2. 关注模型 API 调用次数，每任务应控制在 5-8 次内")
    print(f"  3. 对话压缩机制应在 Extreme 任务中生效，当前 Easy 任务不应触发")
    print(f"  4. 建议在 Docker 环境中进行与竞赛一致的环境验证")
    print(f"\n{'='*80}\n")

    report_path = output_dir / "performance_report.json"
    report_data = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "test_config": "configs/dev.yaml",
            "task_count": len(results),
        },
        "summary": {
            "total_elapsed_seconds": round(total_elapsed, 2),
            "success_rate_pct": round(success_rate, 2),
            "throughput_tasks_per_hour": round(throughput, 2),
            "total_api_calls": total_api_calls,
            "total_errors": total_errors,
            "average_quality_score": round(avg_quality, 1),
            "average_performance_score": round(avg_perf, 1),
            "average_efficiency_score": round(avg_eff, 1),
            "weighted_total_score": round(weighted_total, 1),
            "grade": grade,
            "peak_cpu_pct": round(max(all_peak_cpu), 2),
            "peak_memory_mb": round(max(all_peak_mem), 2),
            "bottlenecks": bottlenecks if bottlenecks else ["None detected"],
        },
        "tasks": task_reports,
    }
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n")
    print(f"  详细报告已保存至: {report_path}")


def main():
    task_dirs = [
        Path(r"D:\2026-KDD-Cup\public\input\task_11"),
        Path(r"D:\2026-KDD-Cup\public\input\task_19"),
        Path(r"D:\2026-KDD-Cup\public\input\task_22"),
        Path(r"D:\2026-KDD-Cup\public\input\task_24"),
        Path(r"D:\2026-KDD-Cup\public\input\task_25"),
    ]

    config_path = Path(r"D:\2026-KDD-Cup\kddcup2026-data-agents-starter-kit\configs\dev.yaml")

    print("=" * 80)
    print("     KDD Cup 2026 Data Agent - 全面性能测试")
    print("=" * 80)
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  测试任务: {', '.join(d.name for d in task_dirs)}")
    print("=" * 80)

    results = run_performance_test(task_dirs, config_path)

    output_dir = Path(r"D:\2026-KDD-Cup\kddcup2026-data-agents-starter-kit\artifacts\perf_tests")
    generate_report(results, output_dir)


if __name__ == "__main__":
    main()
