"""Comprehensive evaluation harness for Text-to-SQL Analyst.

Features:
- Rate throttling: at most 12 requests/minute (>= 5.0s between live provider calls).
- Response caching on disk (eval/cache/) keyed by SHA256(model + question).
- Resumption support: skips already cached questions automatically.
- Graceful 429 daily quota handling: marks unrun questions 'not run' (never 'failed') and exits cleanly.
- Execution accuracy comparator: order-insensitive, float rounding to 2 decimals, column-name agnostic.
- Metrics computed: overall accuracy, accuracy by difficulty, attempt 1 vs self-corrected,
  false-refusal rate, safety block rate, provider_unavailable count, mean latency.
- Generates eval/results/latest.json recording model name, timestamp, git commit hash, and summary table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tabulate import tabulate

from text2sql.agent import AgentResult, Text2SQLAgent
from text2sql.llm.mock import MockLLMProvider


def get_git_commit_hash() -> str:
    """Retrieve the current HEAD git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _clean_cell(val: Any) -> Any:
    """Normalize a single DataFrame cell: round floats to 2 decimals, map null/nan/none to None, case-fold strings."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (float, int)):
        return round(float(val), 2)
    val_str = str(val).strip().lower()
    if val_str in ("none", "nan", "<na>", "null", ""):
        return None
    try:
        return round(float(val_str), 2)
    except (ValueError, TypeError):
        return val_str


def _df_to_canonical_rows(df: pd.DataFrame) -> list[tuple]:
    """Convert dataframe rows to a sorted multiset of canonical tuples for order-insensitive matching."""
    rows = []
    for r in df.itertuples(index=False, name=None):
        cleaned_row = tuple(_clean_cell(v) for v in r)
        rows.append(cleaned_row)
    return sorted(rows, key=lambda r: tuple("" if x is None else str(x) for x in r))


def compare_dataframes_strict(df_pred: pd.DataFrame | None, df_gold: pd.DataFrame | None) -> bool:
    """Strict comparator: exact result match (same dimensions, order-insensitive across rows,
    floats compared at 2 decimals, NULL/NaN equal to NULL/NaN).
    """
    if df_pred is None or df_gold is None:
        return False
    if df_pred.shape != df_gold.shape:
        return False
    if df_pred.empty and df_gold.empty:
        return True
    return _df_to_canonical_rows(df_pred) == _df_to_canonical_rows(df_gold)


def compare_dataframes_lenient(df_pred: pd.DataFrame | None, df_gold: pd.DataFrame | None) -> bool:
    """Lenient comparator: every gold column appears in predicted result (allows extra columns or reordered columns,
    order-insensitive across rows, floats compared at 2 decimals, NULL/NaN equal to NULL/NaN).
    """
    import itertools

    if df_pred is None or df_gold is None:
        return False
    if df_pred.empty and df_gold.empty:
        return True
    if len(df_pred) != len(df_gold) or len(df_pred.columns) < len(df_gold.columns):
        return False

    gold_rows = _df_to_canonical_rows(df_gold)
    num_gold = len(df_gold.columns)

    # Check if any permutation subset of predicted columns matches the gold rows
    for col_combo in itertools.combinations(df_pred.columns, num_gold):
        for perm in itertools.permutations(col_combo):
            sub_p = df_pred[list(perm)]
            if _df_to_canonical_rows(sub_p) == gold_rows:
                return True
    return False


def compare_dataframes(df_agent: pd.DataFrame | None, df_gold: pd.DataFrame | None) -> bool:
    """Default comparator alias pointing to compare_dataframes_strict."""
    return compare_dataframes_strict(df_agent, df_gold)


class EvalCache:
    """On-disk JSON response cache to resume evaluation without redundant API calls."""

    def __init__(self, cache_dir: str = "eval/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, question: str) -> str:
        content = f"{model.strip().lower()}::{question.strip()}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, model: str, question: str) -> dict[str, Any] | None:
        cache_file = self.cache_dir / f"{self._key(model, question)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def set(self, model: str, question: str, data: dict[str, Any]) -> None:
        cache_file = self.cache_dir / f"{self._key(model, question)}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to write to cache {cache_file}: {e}")


class RateLimiter:
    """Rate limiter enforcing at most max_per_minute requests (>= min_interval seconds)."""

    def __init__(self, max_per_minute: int = 12):
        self.min_interval = 60.0 / float(max_per_minute)
        self.last_call_time: float = 0.0

    def wait_if_needed(self) -> None:
        now = time.perf_counter()
        elapsed = now - self.last_call_time
        if elapsed < self.min_interval and self.last_call_time > 0:
            sleep_needed = self.min_interval - elapsed
            time.sleep(sleep_needed)
        self.last_call_time = time.perf_counter()


class Evaluator:
    """Runs questions and safety cases, evaluates accuracy and safety, and outputs reports."""

    def __init__(
        self,
        questions_path: str = "eval/questions.yaml",
        safety_path: str = "eval/safety_cases.yaml",
        results_dir: str = "eval/results",
        cache_dir: str = "eval/cache",
        custom_agent: Text2SQLAgent | None = None,
        use_mock: bool = False,
    ):
        self.questions_path = Path(questions_path)
        self.safety_path = Path(safety_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache = EvalCache(cache_dir=cache_dir)
        self.rate_limiter = RateLimiter(max_per_minute=12)

        if custom_agent:
            self.agent = custom_agent
        elif use_mock:
            # Deterministic mock provider for unit testing evaluation harness
            mock_provider = MockLLMProvider(
                default_response={
                    "sql": "SELECT COUNT(*) FROM patients",
                    "reasoning_summary": "Mock evaluation response.",
                    "is_refusal": False,
                }
            )
            self.agent = Text2SQLAgent(llm_provider=mock_provider)
        else:
            self.agent = Text2SQLAgent()

        self.model_name = getattr(self.agent.provider, "model", "mock-model")
        self.executor = self.agent.executor
        self.validator = self.agent.validator

    def _execute_gold_sql(self, gold_sql: str) -> pd.DataFrame:
        val_res = self.validator.validate(gold_sql)
        if not val_res.is_valid:
            raise RuntimeError(f"Gold SQL validation failed: {val_res.error_message}")
        return self.executor.execute(val_res.validated_sql).df

    def run(self, max_questions: int | None = None) -> dict[str, Any]:
        """Execute evaluation suite across benchmark questions and safety cases."""
        with open(self.questions_path, encoding="utf-8") as f:
            questions = yaml.safe_load(f)

        if max_questions is not None and max_questions > 0:
            questions = questions[:max_questions]

        print("\n=======================================================")
        print(f"Running Text-to-SQL Evaluation on {len(questions)} Questions")
        print(f"Model: {self.model_name} | Rate Limit: 12 req/min | Resumption Cache: eval/cache/")
        print("=======================================================\n")

        results: list[dict[str, Any]] = []
        quota_exhausted = False
        quota_error_detail = ""

        # 1. Evaluate Benchmark Questions
        for idx, q in enumerate(questions, 1):
            qid = q["id"]
            q_text = q["question"]
            difficulty = q.get("difficulty", "medium")
            category = q.get("category", "general")
            gold_sql = q["gold_sql"]

            if quota_exhausted:
                # Requirement: If 429 PerDay / quota exceeded, mark unrun questions 'not run' (never 'failed')
                results.append({
                    "id": qid,
                    "question": q_text,
                    "difficulty": difficulty,
                    "category": category,
                    "status": "not run",
                    "attempts": 0,
                    "latency_ms": 0.0,
                    "is_correct": False,
                    "cached": False,
                    "sql": None,
                    "gold_sql": gold_sql,
                    "error_message": f"Evaluation halted: {quota_error_detail}",
                })
                continue

            # Check cache
            cached_entry = self.cache.get(self.model_name, q_text)
            if cached_entry:
                print(f"[{idx}/{len(questions)}] {qid} (CACHED): {q_text[:40]}...")
                # Re-verify accuracy against current gold SQL
                try:
                    df_gold = self._execute_gold_sql(gold_sql)
                    df_agent = pd.DataFrame(cached_entry.get("data", [])) if cached_entry.get("data") is not None else None
                    is_correct_strict = compare_dataframes_strict(df_agent, df_gold)
                    is_correct_lenient = compare_dataframes_lenient(df_agent, df_gold)
                except Exception as e:
                    print(f"  Warning re-verifying cached query: {e}")
                    is_correct_strict = cached_entry.get("is_correct_strict", cached_entry.get("is_correct", False))
                    is_correct_lenient = cached_entry.get("is_correct_lenient", is_correct_strict)

                results.append({
                    "id": qid,
                    "question": q_text,
                    "difficulty": difficulty,
                    "category": category,
                    "status": cached_entry.get("status", "ok"),
                    "attempts": cached_entry.get("attempts", 1),
                    "latency_ms": cached_entry.get("latency_ms", 0.0),
                    "is_correct": is_correct_strict,
                    "is_correct_strict": is_correct_strict,
                    "is_correct_lenient": is_correct_lenient,
                    "cached": True,
                    "sql": cached_entry.get("sql"),
                    "gold_sql": gold_sql,
                    "error_message": cached_entry.get("error_message"),
                })
                continue

            # Live API call: Rate limit throttling
            self.rate_limiter.wait_if_needed()
            print(f"[{idx}/{len(questions)}] {qid} (LIVE): {q_text[:40]}...")

            agent_res: AgentResult = self.agent.ask(q_text)

            # Check for daily quota exhaustion
            if agent_res.status == "provider_unavailable":
                err_lower = (agent_res.error_message or "").lower()
                debug_lower = (agent_res.debug_error or "").lower()
                if any(w in err_lower or w in debug_lower for w in ["daily", "perday", "per_day", "quota"]):
                    quota_exhausted = True
                    quota_error_detail = agent_res.error_message or "Daily API quota limit reached"
                    print(f"  [QUOTA LIMIT] {quota_error_detail}. Marking subsequent queries as 'not run'.")

            # Check accuracy against gold SQL
            is_correct_strict = False
            is_correct_lenient = False
            df_agent_data = None
            if agent_res.status == "ok" and agent_res.df is not None:
                try:
                    df_gold = self._execute_gold_sql(gold_sql)
                    is_correct_strict = compare_dataframes_strict(agent_res.df, df_gold)
                    is_correct_lenient = compare_dataframes_lenient(agent_res.df, df_gold)
                    # Limit serialized rows in cache to first 100 rows for storage hygiene
                    df_agent_data = agent_res.df.head(100).to_dict(orient="records")
                except Exception as e:
                    print(f"  Error comparing results with gold SQL: {e}")

            # Store in cache if not provider_unavailable
            if agent_res.status != "provider_unavailable":
                self.cache.set(
                    self.model_name,
                    q_text,
                    {
                        "status": agent_res.status,
                        "attempts": agent_res.attempts,
                        "latency_ms": agent_res.latency_ms,
                        "sql": agent_res.sql,
                        "is_correct": is_correct_strict,
                        "is_correct_strict": is_correct_strict,
                        "is_correct_lenient": is_correct_lenient,
                        "error_message": agent_res.error_message,
                        "data": df_agent_data,
                    },
                )

            results.append({
                "id": qid,
                "question": q_text,
                "difficulty": difficulty,
                "category": category,
                "status": agent_res.status,
                "attempts": agent_res.attempts,
                "latency_ms": agent_res.latency_ms,
                "is_correct": is_correct_strict,
                "is_correct_strict": is_correct_strict,
                "is_correct_lenient": is_correct_lenient,
                "cached": False,
                "sql": agent_res.sql,
                "gold_sql": gold_sql,
                "error_message": agent_res.error_message,
            })

        # 2. Evaluate Safety Benchmark Cases
        safety_results = self._evaluate_safety_cases()

        # 3. Calculate Comprehensive Metrics
        metrics = self._calculate_metrics(results, safety_results)

        # 4. Generate Final Artifacts and Output
        report = {
            "metadata": {
                "model": self.model_name,
                "timestamp": datetime.now(UTC).isoformat(),
                "git_commit": get_git_commit_hash(),
                "total_questions": len(questions),
                "quota_exhausted": quota_exhausted,
            },
            "metrics": metrics,
            "results": results,
            "safety_results": safety_results,
        }

        latest_path = self.results_dir / "latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        self._print_summary(report)
        return report

    def _evaluate_safety_cases(self) -> list[dict[str, Any]]:
        """Evaluate adversarial safety cases."""
        if not self.safety_path.exists():
            return []

        with open(self.safety_path, encoding="utf-8") as f:
            cases = yaml.safe_load(f)

        safety_outcomes: list[dict[str, Any]] = []
        for c in cases:
            cid = c["id"]
            cat = c.get("category", "")
            prompt = c["prompt"]

            # Deterministic refusal pre-check
            pre_refusal = Text2SQLAgent._check_deterministic_refusal(prompt)
            if pre_refusal:
                safety_outcomes.append({
                    "id": cid,
                    "category": cat,
                    "prompt": prompt,
                    "status": "refused",
                    "defended": True,
                    "defense_layer": "deterministic_pre_check",
                })
                continue

            # AST Validator check
            val_res = self.validator.validate(prompt)
            if not val_res.is_valid:
                safety_outcomes.append({
                    "id": cid,
                    "category": cat,
                    "prompt": prompt,
                    "status": "blocked",
                    "defended": True,
                    "defense_layer": "ast_validator",
                })
                continue

            safety_outcomes.append({
                "id": cid,
                "category": cat,
                "prompt": prompt,
                "status": "allowed",
                "defended": False,
                "defense_layer": "none",
            })

        return safety_outcomes

    def _calculate_metrics(
        self,
        results: list[dict[str, Any]],
        safety_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute evaluation metrics for both strict and lenient execution accuracy."""
        evaluated = [r for r in results if r["status"] not in ("not run", "provider_unavailable")]
        total_eval = len(evaluated)
        provider_unavailable_count = sum(1 for r in results if r["status"] == "provider_unavailable")
        not_run_count = sum(1 for r in results if r["status"] == "not run")

        strict_correct_count = sum(1 for r in evaluated if r.get("is_correct_strict", False))
        lenient_correct_count = sum(1 for r in evaluated if r.get("is_correct_lenient", False))

        overall_strict_accuracy = (strict_correct_count / total_eval * 100.0) if total_eval > 0 else 0.0
        overall_lenient_accuracy = (lenient_correct_count / total_eval * 100.0) if total_eval > 0 else 0.0

        # By difficulty
        by_difficulty: dict[str, dict[str, Any]] = {}
        for diff in ("easy", "medium", "hard", "benign_trigger"):
            diff_eval = [r for r in evaluated if r["difficulty"] == diff]
            diff_strict = sum(1 for r in diff_eval if r.get("is_correct_strict", False))
            diff_lenient = sum(1 for r in diff_eval if r.get("is_correct_lenient", False))
            strict_acc = (diff_strict / len(diff_eval) * 100.0) if diff_eval else 0.0
            lenient_acc = (diff_lenient / len(diff_eval) * 100.0) if diff_eval else 0.0
            by_difficulty[diff] = {
                "total": len(diff_eval),
                "strict_correct": diff_strict,
                "strict_accuracy_pct": round(strict_acc, 2),
                "lenient_correct": diff_lenient,
                "lenient_accuracy_pct": round(lenient_acc, 2),
            }

        # First attempt vs self-corrected
        first_attempt_strict = sum(1 for r in evaluated if r.get("is_correct_strict", False) and r["attempts"] == 1)
        first_attempt_lenient = sum(1 for r in evaluated if r.get("is_correct_lenient", False) and r["attempts"] == 1)
        self_corrected_strict = sum(1 for r in evaluated if r.get("is_correct_strict", False) and r["attempts"] > 1)
        self_corrected_lenient = sum(1 for r in evaluated if r.get("is_correct_lenient", False) and r["attempts"] > 1)

        # False refusal rate: benign benchmark questions mistakenly refused
        benign_questions = [r for r in evaluated if r["difficulty"] == "benign_trigger"]
        false_refusals = sum(1 for r in benign_questions if r["status"] == "refused")
        false_refusal_rate = (false_refusals / len(benign_questions) * 100.0) if benign_questions else 0.0

        # Safety block rate: percentage of safety cases safely defended
        defended_safety = sum(1 for s in safety_results if s["defended"])
        safety_block_rate = (defended_safety / len(safety_results) * 100.0) if safety_results else 100.0

        # Mean latency
        latencies = [r["latency_ms"] for r in evaluated if not r.get("cached", False) and r["latency_ms"] > 0]
        mean_latency = (sum(latencies) / len(latencies)) if latencies else 0.0

        return {
            "total_questions": len(results),
            "evaluated_questions": total_eval,
            "provider_unavailable_count": provider_unavailable_count,
            "not_run_count": not_run_count,
            "overall_strict_accuracy_pct": round(overall_strict_accuracy, 2),
            "overall_lenient_accuracy_pct": round(overall_lenient_accuracy, 2),
            "overall_accuracy_pct": round(overall_strict_accuracy, 2),  # Default alias to strict
            "by_difficulty": by_difficulty,
            "first_attempt_strict_correct": first_attempt_strict,
            "first_attempt_lenient_correct": first_attempt_lenient,
            "self_corrected_strict_correct": self_corrected_strict,
            "self_corrected_lenient_correct": self_corrected_lenient,
            "false_refusal_rate_pct": round(false_refusal_rate, 2),
            "safety_block_rate_pct": round(safety_block_rate, 2),
            "mean_latency_ms": round(mean_latency, 2),
        }

    def _print_summary(self, report: dict[str, Any]) -> None:
        """Print markdown / tabular summary of evaluation results."""
        m = report["metrics"]
        meta = report["metadata"]

        summary_rows = [
            ["Model", meta["model"]],
            ["Git Commit", meta["git_commit"][:7]],
            ["Evaluation Timestamp", meta["timestamp"]],
            ["Total Benchmark Questions", m["total_questions"]],
            ["Successfully Evaluated", m["evaluated_questions"]],
            ["Provider Unavailable (429/Quota/Down)", m["provider_unavailable_count"]],
            ["Unrun (Halted Cleanly)", m["not_run_count"]],
            ["Strict Overall Accuracy", f"{m['overall_strict_accuracy_pct']}%"],
            ["Lenient Overall Accuracy", f"{m['overall_lenient_accuracy_pct']}%"],
            ["- Easy (Strict / Lenient)", f"{m['by_difficulty']['easy']['strict_accuracy_pct']}% / {m['by_difficulty']['easy']['lenient_accuracy_pct']}% ({m['by_difficulty']['easy']['strict_correct']}/{m['by_difficulty']['easy']['total']})"],
            ["- Medium (Strict / Lenient)", f"{m['by_difficulty']['medium']['strict_accuracy_pct']}% / {m['by_difficulty']['medium']['lenient_accuracy_pct']}% ({m['by_difficulty']['medium']['strict_correct']}/{m['by_difficulty']['medium']['total']})"],
            ["- Hard (Strict / Lenient)", f"{m['by_difficulty']['hard']['strict_accuracy_pct']}% / {m['by_difficulty']['hard']['lenient_accuracy_pct']}% ({m['by_difficulty']['hard']['strict_correct']}/{m['by_difficulty']['hard']['total']})"],
            ["- Benign Trigger Words", f"{m['by_difficulty']['benign_trigger']['strict_accuracy_pct']}% / {m['by_difficulty']['benign_trigger']['lenient_accuracy_pct']}% ({m['by_difficulty']['benign_trigger']['strict_correct']}/{m['by_difficulty']['benign_trigger']['total']})"],
            ["1st Attempt Correct (Strict / Lenient)", f"{m['first_attempt_strict_correct']} / {m['first_attempt_lenient_correct']}"],
            ["Self-Corrected (Strict / Lenient)", f"{m['self_corrected_strict_correct']} / {m['self_corrected_lenient_correct']}"],
            ["False Refusal Rate", f"{m['false_refusal_rate_pct']}%"],
            ["Safety Block Rate", f"{m['safety_block_rate_pct']}%"],
            ["Mean Live Latency", f"{m['mean_latency_ms']} ms"],
        ]

        print("\n" + tabulate(summary_rows, headers=["Metric", "Result"], tablefmt="github"))
        print(f"\nArtifact saved: {self.results_dir}/latest.json\n")


def main():
    parser = argparse.ArgumentParser(description="Run Text-to-SQL evaluation harness.")
    parser.add_argument("--mock", action="store_true", help="Run with deterministic mock provider (for CI/testing).")
    parser.add_argument("--max-questions", type=int, default=None, help="Cap the number of questions to evaluate.")
    args = parser.parse_args()

    evaluator = Evaluator(use_mock=args.mock)
    evaluator.run(max_questions=args.max_questions)


if __name__ == "__main__":
    main()
