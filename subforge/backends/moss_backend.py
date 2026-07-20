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

import threading
from typing import Any, Iterable

from ..config import ModelSpec
from ..transcriber import Segment, TranscriptResult


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
            import os as _os

            local_only = init.get("local_files_only", _os.environ.get("MOSS_ALLOW_DOWNLOAD") != "1")
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

        device = resolve_device("auto")
        dtype_name = gen.get("dtype") or (self.spec.init or {}).get("dtype") or "bfloat16"
        dtype = getattr(torch, dtype_name) if isinstance(dtype_name, str) else dtype_name

        messages = build_transcription_messages(audio)
        result = generate_transcription(
            self._model,
            self._processor,
            messages,
            max_new_tokens=gen.get("max_new_tokens", 65536),
            do_sample=gen.get("do_sample", False),
            device=device,
            dtype=dtype,
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