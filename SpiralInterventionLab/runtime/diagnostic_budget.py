"""Episode accounting must not depend on the size of a retained evidence window."""
from typing import Any


def used(worker: Any) -> int:
    retained_cost = sum(
        int(row.get("diagnostic_call_cost", bool(row.get("diagnostic_budget_charged", True))))
        for row in getattr(worker, "_diagnostic_results", ())
    )
    return max(int(getattr(worker, "_diagnostic_calls_used", 0)),
               retained_cost)


def left(worker: Any) -> int:
    return max(0, worker.max_diagnostic_calls_per_run - used(worker))
