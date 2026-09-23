"""Episode accounting must not depend on the size of a retained evidence window."""
from typing import Any


def used(worker: Any) -> int:
    return max(int(getattr(worker, "_diagnostic_calls_used", 0)),
               len(getattr(worker, "_diagnostic_results", ())))


def left(worker: Any) -> int:
    return max(0, worker.max_diagnostic_calls_per_run - used(worker))
