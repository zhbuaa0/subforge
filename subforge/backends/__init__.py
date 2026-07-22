"""ASR model backends.

Each backend wraps a different model-loading library (FunASR / HuggingFace
transformers / ...) behind a single ``transcribe(audio, **kwargs) -> list[TranscriptResult]``
contract.  ``ModelRegistry`` dispatches based on ``spec.backend``.

Public surface:
    BaseBackend         — typing protocol
    FunasrBackend       — wraps funasr.AutoModel (default; paraformer-*, sensevoice)
    MossBackend         — moss_transcribe_diarize + transformers (moss-transcribe-diarize)
    MossVllmBackend     — Moss via OpenAI compat forward to a vLLM server
"""

from __future__ import annotations

from typing import Union

from .base import BaseBackend
from .funasr_backend import FunasrBackend
from .moss_backend import MossBackend
from .vllm_backend import MossVllmBackend

Backend = Union[FunasrBackend, MossBackend, MossVllmBackend]

__all__ = ["Backend", "BaseBackend", "FunasrBackend", "MossBackend", "MossVllmBackend"]