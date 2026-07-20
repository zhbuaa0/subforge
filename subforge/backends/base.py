"""Common backend protocol for all ASR model adapters.

Both FunASR and MOSS adapters implement this same shape so that
``Transcriber`` / ``cli.py`` / ``server.py`` can stay backend-agnostic.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from ..config import ModelSpec
from ..transcriber import TranscriptResult


@runtime_checkable
class BaseBackend(Protocol):
    """A loaded ASR model, ready to transcribe.

    Attributes:
        name:  Model name as registered in models.yaml.
        spec:  The ModelSpec that produced this backend.
    """

    name: str
    spec: ModelSpec

    def transcribe(
        self,
        audio: str | Iterable[str],
        **kwargs: Any,
    ) -> list[TranscriptResult]:
        """Run inference; return one TranscriptResult per input audio.

        Backend implementations are responsible for normalizing their raw
        output into subforge's ``TranscriptResult`` schema (with ``Segment``
        entries carrying ``start`` / ``end`` / ``text`` / ``spk``).

        Args:
            audio: A single audio path/URL, or an iterable of multiple.
            **kwargs: Backend-specific overrides merged on top of spec.generate.
        """
        ...