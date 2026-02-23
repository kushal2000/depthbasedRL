"""Lightweight CUDA-event based profiler for RL training loops."""

import torch


class CudaTimer:
    """Accumulates GPU timings using CUDA events.

    Usage:
    
        timer = CudaTimer(device)
        timer.start('section_name')
        ... gpu work ...
        timer.stop('section_name')

        # After epoch:
        results = timer.flush()  # returns {name: elapsed_ms}
    """

    def __init__(self, device, enabled=True):
        self.device = device
        self.enabled = enabled
        self._starts: dict = {}
        self._accum: dict = {}  # name -> list of (start_event, end_event)

    def start(self, name: str):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream(self.device))
        self._starts[name] = ev

    def stop(self, name: str):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(torch.cuda.current_stream(self.device))
        start_ev = self._starts.pop(name, None)
        if start_ev is not None:
            self._accum.setdefault(name, []).append((start_ev, ev))

    def flush(self) -> dict:
        """Synchronize and return accumulated timings in milliseconds."""
        if not self.enabled:
            return {}
        torch.cuda.synchronize(self.device)
        results = {}
        for name, pairs in self._accum.items():
            total_ms = sum(s.elapsed_time(e) for s, e in pairs)
            results[name] = total_ms
        self._accum.clear()
        self._starts.clear()
        return results
