"""MOSS backend — wraps HuggingFace transformers + moss_transcribe_diarize.

Supports long-form multi-speaker transcription with built-in timestamps and
diarization (no separate VAD / punc / spk sub-models needed).  Loads the HF
weights lazily on first ``transcribe()``; caches both ``model`` and
``processor`` on the backend.

Differences from the FunASR backend that callers must be aware of:

- Single audio per call: ``MossBackend.transcribe(audio, ...)`` raises
  ``ValueError`` if ``audio`` is an iterable with more than one entry.
- Speaker labels arrive as ``"S01"`` / ``"S02"`` strings; we map them to
  ints in first-appearance order so the rest of subforge (which uses
  ``Segment.spk: int``) works unchanged.
- ``preset_spk_num`` is silently dropped (MOSS detects speakers on its own).
- Streaming is not supported — ``spec.streaming`` must be ``False``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Callable, Iterable

from ..config import ModelSpec
from ..transcriber import Segment, TranscriptResult


# stderr logger that the caller can mute via env var
def _log(msg: str) -> None:
    if os.environ.get("SUBFORGE_QUIET") == "1":
        return
    print(msg, file=sys.stderr, flush=True)


class _ProgressPrinter:
    """Throttled progress reporter for MOSS decoding.

    Receives ``token_callback``-style updates (cumulative generated token count)
    from ``moss_transcribe_diarize.inference_utils.generate_transcription`` and
    prints a single-line progress message every ``min_interval_s`` seconds so
    the user can see inference is alive on long-form audio (MOSS can take
    30+ minutes for 11-min audio with the default HF backend).

    Also forwards 3-arg ``(phase, pct, msg)`` updates to an external callback
    (used by the FastAPI server to push progress over SSE).

    Output to stderr is suppressed when stdout/stderr is not a TTY (e.g. piping
    JSON into another tool), or when ``SUBFORGE_QUIET=1`` is set.
    """

    def __init__(
        self,
        max_new_tokens: int,
        *,
        min_interval_s: float = 10.0,
        log_every_tokens: int = 200,
        external: Callable[[str, float, str], None] | None = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.min_interval_s = min_interval_s
        self.log_every_tokens = log_every_tokens
        self._external = external
        self._start: float | None = None
        self._last_print_t: float = 0.0
        self._last_printed_count: int = 0
        self._last_external_count: int = -1
        self._silent = not sys.stderr.isatty()

    def __call__(self, generated_tokens: int) -> None:
        # Always notify external listener (SSE) on token change, but throttle
        if (
            self._external is not None
            and generated_tokens != self._last_external_count
            and generated_tokens - (self._last_external_count or 0) >= 50
        ):
            self._last_external_count = generated_tokens
            pct = (
                generated_tokens / self.max_new_tokens
                if self.max_new_tokens
                else 0.0
            )
            try:
                self._external(
                    "decoding",
                    min(pct, 0.99),
                    f"{generated_tokens}/{self.max_new_tokens} tokens",
                )
            except Exception:  # noqa: BLE001
                pass

        if self._silent:
            return
        now = time.monotonic()
        if self._start is None:
            self._start = now
            self._last_print_t = now
        # Throttle by time AND by token count (whichever fires first)
        elapsed = now - self._start
        if (now - self._last_print_t) < self.min_interval_s and (
            generated_tokens - self._last_printed_count
        ) < self.log_every_tokens:
            return
        rate = generated_tokens / elapsed if elapsed > 0 else 0.0
        pct = (
            100.0 * generated_tokens / self.max_new_tokens
            if self.max_new_tokens
            else 0.0
        )
        eta = (
            (self.max_new_tokens - generated_tokens) / rate
            if rate > 0 and self.max_new_tokens
            else float("nan")
        )
        # Use \r so consecutive prints overwrite the same line in interactive shells
        _log(
            f"\r[MOSS] {generated_tokens}/{self.max_new_tokens} tokens "
            f"({pct:5.1f}%) | elapsed {elapsed:6.1f}s | "
            f"{rate:5.2f} tok/s | ETA {eta:5.0f}s   "
        )
        self._last_print_t = now
        self._last_printed_count = generated_tokens


class MossBackend:
    """A lazy-loaded MOSS-Transcribe-Diarize model + processor."""

    def __init__(self, spec: ModelSpec):
        self.name = spec.name
        self.spec = spec
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        """Load HF transformers model + processor. Idempotent; thread-safe."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

            init = self.spec.init or {}
            dtype_name = init.get("dtype", "bfloat16")
            dtype = getattr(torch, dtype_name) if isinstance(dtype_name, str) else dtype_name
            # Default to offline: HF cache lookup first (no network). Set
            # ``MOSS_ALLOW_DOWNLOAD=1`` to opt back into Hub downloads.
            local_only = init.get("local_files_only", os.environ.get("MOSS_ALLOW_DOWNLOAD") != "1")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.model,
                trust_remote_code=init.get("trust_remote_code", True),
                dtype=dtype,
                device_map=init.get("device_map", "cuda:0"),
                local_files_only=local_only,
            ).eval()
            self._processor = AutoProcessor.from_pretrained(
                self.spec.model,
                trust_remote_code=True,
                local_files_only=local_only,
            )

    @property
    def model(self) -> Any:
        self._load()
        return self._model

    @property
    def processor(self) -> Any:
        self._load()
        return self._processor

    def transcribe(
        self,
        audio: str | Iterable[str],
        *,
        progress: Callable[[str, float, str], None] | None = None,
        **kwargs: Any,
    ) -> list[TranscriptResult]:
        if isinstance(audio, (list, tuple)):
            if len(audio) == 0:
                raise ValueError("MossBackend.transcribe: 空音频列表")
            if len(audio) > 1:
                raise ValueError(
                    "MossBackend 只支持单音频输入；如需批处理请改用 FunASR backend 或外层循环"
                )
            audio = audio[0]  # type: ignore[assignment]
        if not isinstance(audio, str):
            raise TypeError(f"MossBackend.transcribe: 期望 str，得到 {type(audio).__name__}")

        self._load()

        # Defer moss imports until we actually need them so the FunASR-only
        # code path doesn't pay the transformers / torch import cost.
        from moss_transcribe_diarize import parse_transcript  # type: ignore
        from moss_transcribe_diarize.inference_utils import (  # type: ignore
            build_transcription_messages,
            generate_transcription,
            resolve_device,
        )
        import torch

        gen: dict[str, Any] = dict(self.spec.generate)
        gen.update(kwargs)
        gen.pop("preset_spk_num", None)  # MOSS 自动检测,忽略此参数
        gen.pop("progress", None)  # not a generate kwarg

        device = resolve_device("auto")
        dtype_name = gen.get("dtype") or (self.spec.init or {}).get("dtype") or "bfloat16"
        dtype = getattr(torch, dtype_name) if isinstance(dtype_name, str) else dtype_name

        max_new_tokens = int(gen.get("max_new_tokens", 65536))

        # Push external progress (SSE) via the printer's external callback.
        # SSE phase "decoding" uses token progress; we wrap it so the server
        # gets nice 0-99% updates as MOSS generates tokens.
        printer = _ProgressPrinter(max_new_tokens=max_new_tokens, external=progress)
        if progress is not None:
            try:
                progress("loading", 0.02, "model loaded")
            except Exception:  # noqa: BLE001
                pass

        messages = build_transcription_messages(audio)
        _log(
            f"[MOSS] starting inference (max_new_tokens={max_new_tokens}, "
            f"do_sample={gen.get('do_sample', False)})"
        )
        try:
            result = generate_transcription(
                self._model,
                self._processor,
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=gen.get("do_sample", False),
                device=device,
                dtype=dtype,
                token_callback=printer,
            )
        finally:
            # Force the final newline so the next [INFO] line starts fresh
            if not printer._silent:
                print(file=sys.stderr, flush=True)
        _log(
            f"[MOSS] inference finished: {result.get('generated_tokens', '?')} tokens "
            f"in {result.get('prompt_len', '?')} prompt tokens"
        )

        raw_text = result["text"]
        moss_segs = parse_transcript(raw_text)

        # MOSS speaker strings → stable int IDs in first-appearance order
        spk_map: dict[str, int] = {}
        for s in moss_segs:
            if s.speaker not in spk_map:
                spk_map[s.speaker] = len(spk_map)

        sub_segs = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                spk=spk_map.get(s.speaker),
            )
            for s in moss_segs
        ]

        return [
            TranscriptResult(
                text="".join(s.text for s in sub_segs),
                segments=sub_segs,
                num_speakers=len(spk_map) if spk_map else None,
                language=None,
                # raw 留 None：subforge.exporters.to_json() 检测到 raw 缺失时会回退到
                # funasr 形状的 sentence_info 序列化，保证 ``asr export`` / 下游消费脚本
                # 拿到统一 schema。MOSS 自身的 prompt_len / generated_tokens 仅 debug 用。
                raw=None,
            )
        ]