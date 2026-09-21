"""Existing CLI with timing-only instrumentation and inference autograd disabled."""
import faulthandler
import json
import time

import torch

from SpiralInterventionLab.examples.digit_transform_e2e import main
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    faulthandler.dump_traceback_later(120, repeat=True)
    step = HookedTransformerWorkerRuntime.step

    def timed_step(self):
        start = time.monotonic()
        result = step(self)
        print(json.dumps({"event": "worker_step_timing", "step": self._steps,
                          "seconds": time.monotonic() - start,
                          "output": self.final_text()}), flush=True)
        return result

    HookedTransformerWorkerRuntime.step = timed_step
    try:
        raise SystemExit(main())
    finally:
        faulthandler.cancel_dump_traceback_later()
