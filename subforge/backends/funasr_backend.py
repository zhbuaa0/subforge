"""FunASR backend — wraps ``funasr.AutoModel`` (the original default).

Lazy-loads the model on first ``transcribe()`` call; the AutoModel instance
is cached on the backend object.  Behaviour is identical to the pre-refactor
``subforge.models.ModelRegistry.get(name)`` + ``subforge.transcriber.Transcriber.transcribe()``
path — this module is the new home for the exact same code.
"""

from __future__ import annotations

import threading
from typing import Any, Iterable

from ..config import ModelSpec
from ..transcriber import TranscriptResult, _apply_postprocess, _normalize


class FunasrBackend:
    """A lazy-loaded ``funasr.AutoModel`` instance plus its spec."""

    def __init__(self, spec: ModelSpec):
        self.name = spec.name
        self.spec = spec
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from funasr import AutoModel  # type: ignore

                    self._model = AutoModel(**self.spec.auto_model_kwargs())
        return self._model

    def transcribe(
        self,
        audio: str | Iterable[str],
        *,
        preset_spk_num: int | None = None,
        **generate_overrides: Any,
    ) -> list[TranscriptResult]:
        """Run FunASR ``AutoModel.generate(...)`` and normalize the output.

        Mirrors the pre-refactor Transcriber behaviour exactly:
        - merges spec.generate with overrides
        - drops ``preset_spk_num`` when None (FunASR auto-detects speakers)
        - applies spec.postprocess if set
        """
        gen_kwargs: dict[str, Any] = dict(self.spec.generate)
        gen_kwargs.update(generate_overrides)
        if preset_spk_num is not None:
            gen_kwargs["preset_spk_num"] = int(preset_spk_num)
        # funasr 不接受 preset_spk_num=None；省略即自动检测（关键！）

        raw = self.model.generate(input=audio, **gen_kwargs)
        results = _normalize(raw or [])
        if results:
            _apply_postprocess(results, self.spec)
        return results