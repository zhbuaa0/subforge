"""ASR model backends.

Each backend wraps a different model-loading library (FunASR / HuggingFace
transformers / ...) behind a single ``transcribe(audio, **kwargs) -> list[TranscriptResult]``
contract.  ``ModelRegistry`` dispatches based on ``spec.backend``.

Public surface:
    BaseBackend       — typing protocol
    FunasrBackend     — wraps funasr.AutoModel (the default; used for paraformer-*, sensevoice)
    MossBackend       — wraps moss_transcribe_diarize + transformers (used for moss-transcribe-diarize)
"""

from __future__ import annotations

from typing import Union

from .base import BaseBackend
from .funasr_backend import FunasrBackend
from .moss_backend import MossBackend

Backend = Union[FunasrBackend, MossBackend]

__all__ = ["Backend", "BaseBackend", "FunasrBackend", "MossBackend"]