"""FunASR backend — wraps ``funasr.AutoModel`` (the original default).

Lazy-loads the model on first ``transcribe()`` call; the AutoModel instance
is cached on the backend object.  Behaviour is identical to the pre-refactor
``subforge.models.ModelRegistry.get(name)`` + ``subforge.transcriber.Transcriber.transcribe()``
path — this module is the new home for the exact same code.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Iterable

from ..config import ModelSpec
from ..transcriber import TranscriptResult, _apply_postprocess, _normalize


def _log(msg: str) -> None:
    if os.environ.get("SUBFORGE_QUIET") == "1":
        return
    print(msg, file=sys.stderr, flush=True)


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
        progress: "ProgressCallback | None" = None,
        **generate_overrides: Any,
    ) -> list[TranscriptResult]:
        """Run FunASR ``AutoModel.generate(...)`` and normalize the output.

        Args:
            progress: Optional callback ``(phase: str, pct: float, msg: str)``.
                FunASR's ``AutoModel.generate`` is a black box, so we report
                coarse phases (``loading_model``, ``vad``, ``asr``, ``spk``,
                ``done``) instead of token-level progress.  MOSS backend has
                fine-grained token callbacks; FunASR doesn't expose them.
        """
        gen_kwargs: dict[str, Any] = dict(self.spec.generate)
        gen_kwargs.update(generate_overrides)
        if preset_spk_num is not None:
            gen_kwargs["preset_spk_num"] = int(preset_spk_num)
        # funasr 不接受 preset_spk_num=None；省略即自动检测（关键！）

        # FunASR 自带 tqdm 进度条到 stderr；在交互终端里开启
        # (非 TTY 场景会被 funasr 自己跳过)
        gen_kwargs.setdefault("disable_tqdm", not sys.stderr.isatty())

        if progress is not None:
            progress("loading_model", 0.05, f"loading {self.spec.model}")

        raw = self.model.generate(input=audio, **gen_kwargs)

        if progress is not None:
            progress("postprocess", 0.95, "normalizing segments")

        results = _normalize(raw or [])
        if results:
            _apply_postprocess(results, self.spec)

        if progress is not None:
            progress("done", 1.0, "complete")

        return results


# 进度回调类型（与 MossBackend 共享签名）
from typing import Callable
ProgressCallback = Callable[[str, float, str], None]