# p2p_tracker.py

from __future__ import annotations
import time
import p2p_vlm


class ExperimentTracker:
    def __init__(self):
        self._start: float | None = None
        self.elapsed: float       = 0.0
        self.input_tokens: int    = 0
        self.output_tokens: int   = 0

    def start(self):
        """실험 시작 — 타이머 + 토큰 카운터 초기화."""
        self._start        = time.time()
        self.elapsed       = 0.0
        self.input_tokens  = 0
        self.output_tokens = 0
        # p2p_vlm의 누적 카운터 초기화
        if hasattr(p2p_vlm, "_total_usage"):
            with p2p_vlm._usage_lock:
                p2p_vlm._total_usage["prompt_tokens"]     = 0
                p2p_vlm._total_usage["completion_tokens"] = 0

    def stop(self):
        """실험 종료 — 경과 시간 + 토큰 수 기록."""
        if self._start is not None:
            self.elapsed = round(time.time() - self._start, 2)
        # p2p_vlm의 누적 카운터 읽기
        if hasattr(p2p_vlm, "_total_usage"):
            with p2p_vlm._usage_lock:
                self.input_tokens  = p2p_vlm._total_usage["prompt_tokens"]
                self.output_tokens = p2p_vlm._total_usage["completion_tokens"]

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self, label: str = "") -> str:
        tag = f"[{label}] " if label else ""
        return (
            f"  {tag}PT = {self.elapsed:.1f}s  |  "
            f"TC = {self.total_tokens:,} tokens "
            f"(in={self.input_tokens:,} / out={self.output_tokens:,})"
        )

    def as_dict(self) -> dict:
        return {
            "pt":            self.elapsed,
            "tc":            self.total_tokens,
            "input_tokens":  self.input_tokens,
            "output_tokens": self.output_tokens,
        }


tracker = ExperimentTracker()
