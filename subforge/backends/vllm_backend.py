"""MOSS backend that talks to an external vLLM server via /v1/audio/transcriptions.

Use this backend when you have a vLLM server running MOSS-Transcribe-Diarize
(``vllm serve OpenMOSS-Team/MOSS-Transcribe-Diarize --trust-remote-code``).
Throughput is 5-10x the local HF transformers path on the same GPU.

What this module does:

- Sends the audio file as multipart form data to
  ``{base_url}/v1/audio/transcriptions`` with ``stream: true``.
  This is vLLM 0.23+'s native transcription endpoint for MOSS.
- Streams SSE chunks (``transcription.chunk`` objects) containing
  ``delta.content`` text deltas and pushes progress to the FastAPI SSE
  pipeline (``progress("decoding", pct, msg)``) and stderr.
- Reuses ``_moss_text_to_result`` from ``moss_backend`` for the final parse
  so downstream code (exporters, REST API) sees identical output to
  ``MossBackend``.

Config comes from ``ModelSpec.init``:

    init:
      base_url: http://127.0.0.1:8001
      api_key_env: VLLM_API_KEY    # optional; empty/unset -> no Authorization
      timeout: 3600                # seconds; long audio can take 30+ minutes
      model_name: OpenMOSS-Team/MOSS-Transcribe-Diarize

And from ``ModelSpec.generate`` (merged with kwargs at call time):

    generate:
      max_new_tokens: 65536
      do_sample: false
      prompt: ...         # optional; used as the `prompt` form field
"""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from ..config import ModelSpec
from ..transcriber import TranscriptResult
from .moss_backend import _moss_text_to_result  # shared parser


# ---------------------------------------------------------------------------
# stderr logger
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    if os.environ.get("SUBFORGE_QUIET") == "1":
        return
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------

class _VllmProgressPrinter:
    """Throttle-prints vLLM token progress to stderr.

    vLLM transcription SSE chunks carry ``delta.content`` text deltas, not
    token counts, so we estimate token count by accumulating character length
    and dividing by a rough chars-per-token ratio (~3.5 for mixed CN/EN/text).
    """

    def __init__(
        self,
        max_new_tokens: int,
        *,
        min_interval_s: float = 10.0,
        min_chars_per_tick: int = 200,
        external: Callable[[str, float, str], None] | None = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.min_interval_s = min_interval_s
        self.min_chars_per_tick = min_chars_per_tick
        self._external = external
        self._start: float | None = None
        self._last_print_t: float = 0.0
        self._accum_chars: int = 0
        self._last_external_chars: int = -1
        self._estimated_tokens: int = 0
        self._silent = not sys.stderr.isatty()

    @property
    def estimated_tokens(self) -> int:
        return self._estimated_tokens

    def feed(self, delta_text: str) -> None:
        if not delta_text:
            return
        self._accum_chars += len(delta_text)
        # Rough estimate: 1 token ~ 3.5 chars for MOSS output (CN-heavy)
        self._estimated_tokens = int(self._accum_chars / 3.5)

        # External (SSE) push — every ~50 tokens worth of chars
        if self._external is not None:
            if (
                self._estimated_tokens != self._last_external_chars
                and self._estimated_tokens - max(self._last_external_chars, 0) >= 50
            ):
                self._last_external_chars = self._estimated_tokens
                pct = (
                    self._estimated_tokens / self.max_new_tokens
                    if self.max_new_tokens
                    else 0.0
                )
                try:
                    self._external(
                        "decoding",
                        min(pct, 0.99),
                        f"{self._estimated_tokens}/{self.max_new_tokens} tokens",
                    )
                except Exception:  # noqa: BLE001
                    pass

        if self._silent:
            return
        now = time.monotonic()
        if self._start is None:
            self._start = now
            self._last_print_t = now
        if (now - self._last_print_t) < self.min_interval_s and (
            self._accum_chars - getattr(self, "_last_print_chars", 0)
        ) < self.min_chars_per_tick:
            return
        elapsed = now - self._start
        rate = self._estimated_tokens / elapsed if elapsed > 0 else 0.0
        pct = (
            100.0 * self._estimated_tokens / self.max_new_tokens
            if self.max_new_tokens
            else 0.0
        )
        eta = (
            (self.max_new_tokens - self._estimated_tokens) / rate
            if rate > 0 and self.max_new_tokens
            else float("nan")
        )
        _log(
            f"\r[vLLM] {self._estimated_tokens}/{self.max_new_tokens} tokens "
            f"(est, {pct:5.1f}%) | elapsed {elapsed:6.1f}s | "
            f"{rate:5.2f} tok/s | ETA {eta:5.0f}s   "
        )
        self._last_print_t = now
        self._last_print_chars = self._accum_chars


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

class _VllmHttpError(RuntimeError):
    """vLLM returned a 4xx/5xx."""

    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"vLLM HTTP {status} from {url}: {body[:500]}")
        self.status = status
        self.body = body
        self.url = url


def _requests_post_streaming(
    url: str,
    fields: dict[str, str],
    file_data: tuple[str, bytes, str],
    headers: dict[str, str],
    timeout: float,
):
    """Use requests to POST multipart form + stream response; raise on HTTP error."""
    import requests

    filename, file_bytes, mime = file_data
    req_headers = {"Accept": "text/event-stream", **headers}

    try:
        resp = requests.post(
            url,
            data=fields,
            files={"file": (filename, file_bytes, mime)},
            headers=req_headers,
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp
    except requests.HTTPError as e:
        try:
            body = e.response.text[:500]
        except Exception:  # noqa: BLE001
            body = str(e)
        raise _VllmHttpError(e.response.status_code, body, url) from e
    except requests.RequestException as e:
        raise _VllmHttpError(0, str(e), url) from e


def _vllm_post_multipart_streaming(
    base_url: str,
    fields: dict[str, str],
    file_path: Path,
    headers: dict[str, str],
    timeout: float,
):
    """POST multipart form to ``{base_url}/v1/audio/transcriptions``; yield SSE chunks."""
    url = f"{base_url.rstrip('/')}/v1/audio/transcriptions"

    mime, _ = mimetypes.guess_type(str(file_path))
    if mime is None:
        mime = "audio/wav"

    # Build files dict: (filename, content, mime_type)
    file_data: tuple[str, bytes, str] = (file_path.name, file_path.read_bytes(), mime)

    try:
        resp = _requests_post_streaming(url, fields, file_data, headers, timeout)
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                decoded = line.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                continue
            if decoded.startswith(":"):
                continue
            if decoded.startswith("data:"):
                payload = decoded[len("data:"):].lstrip()
                if payload.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
    except _VllmHttpError:
        raise
    except Exception as e:
        raise _VllmHttpError(0, str(e), url) from e


# ---------------------------------------------------------------------------
# Main backend class
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


class MossVllmBackend:
    """A no-weight backend that forwards MOSS inference to a vLLM server."""

    def __init__(self, spec: ModelSpec):
        self.name = spec.name
        self.spec = spec
        if spec.streaming:
            raise ValueError(
                f"MossVllmBackend: spec '{spec.name}' 标记为 streaming; "
                "vLLM 不支持流式,请在 models.yaml 把 streaming 置为 false"
            )
        init = spec.init or {}
        self._base_url: str = init.get("base_url", "http://127.0.0.1:8001")
        self._api_key_env: str = init.get("api_key_env", "") or ""
        self._timeout: float = float(init.get("timeout", 3600))
        self._model_name: str = init.get("model_name") or spec.model

    # ------------------------------------------------------------------
    # BaseBackend protocol
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: str | Iterable[str],
        *,
        progress: Callable[[str, float, str], None] | None = None,
        **kwargs: Any,
    ) -> list[TranscriptResult]:
        # ---- input handling ----
        if isinstance(audio, (list, tuple)):
            if len(audio) == 0:
                raise ValueError("MossVllmBackend.transcribe: 空音频列表")
            if len(audio) > 1:
                raise ValueError(
                    "MossVllmBackend 只支持单音频输入；如需批处理请改用 FunASR backend"
                )
            audio = audio[0]  # type: ignore[assignment]
        if not isinstance(audio, str):
            raise TypeError(
                f"MossVllmBackend.transcribe: 期望 str，得到 {type(audio).__name__}"
            )

        audio_path = Path(audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"vllm backend: 音频文件不存在: {audio}")

        # ---- merge generate config ----
        gen: dict[str, Any] = dict(self.spec.generate)
        gen.update(kwargs)
        gen.pop("preset_spk_num", None)
        gen.pop("progress", None)
        base_url = gen.pop("vllm_base_url", None) or self._base_url

        max_new_tokens = int(gen.get("max_new_tokens", 65536))
        do_sample = bool(gen.get("do_sample", False))
        prompt = str(gen.get("prompt") or _DEFAULT_PROMPT)
        temperature = 0.0 if not do_sample else float(gen.get("temperature", 0.7))

        # ---- headers ----
        headers: dict[str, str] = {}
        if self._api_key_env:
            api_key = os.environ.get(self._api_key_env, "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                _log(
                    f"[vLLM] 环境变量 {self._api_key_env} 未设置;"
                    "以无鉴权方式调用 vLLM"
                )

        fields = {
            "model": self._model_name,
            "temperature": str(temperature),
            "prompt": prompt,
            "stream": "true",
        }

        printer = _VllmProgressPrinter(max_new_tokens=max_new_tokens, external=progress)
        if progress is not None:
            try:
                progress("loading", 0.02, "vllm request prepared")
            except Exception:  # noqa: BLE001
                pass

        _log(f"[vLLM] starting inference (model={self._model_name}, url={base_url})")
        try:
            return self._stream_and_parse(
                base_url, fields, audio_path, headers, printer, progress
            )
        finally:
            if not printer._silent:
                print(file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    # Internal: drive the SSE stream and consolidate output
    # ------------------------------------------------------------------

    def _stream_and_parse(
        self,
        base_url: str,
        fields: dict[str, str],
        audio_path: Path,
        headers: dict[str, str],
        printer: _VllmProgressPrinter,
        progress: Callable[[str, float, str], None] | None,
    ) -> list[TranscriptResult]:
        assembled: list[str] = []
        try:
            for chunk in _vllm_post_multipart_streaming(
                base_url, fields, audio_path, headers, self._timeout
            ):
                # vLLM transcription chunk:
                # {"id":"transcribe-xxx","object":"transcription.chunk",
                #  "choices":[{"delta":{"content":"累积文本"}}]}
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        assembled.append(delta)
                        printer.feed(delta)
                        if progress is not None:
                            try:
                                progress(
                                    "decoding",
                                    min(
                                        printer.estimated_tokens
                                        / max(printer.max_new_tokens, 1),
                                        0.99,
                                    ),
                                    f"{printer.estimated_tokens}/{printer.max_new_tokens} tokens",
                                )
                            except Exception:  # noqa: BLE001
                                pass
        except _VllmHttpError as e:
            _log(f"[vLLM] error: {e}")
            raise

        raw_text = "".join(assembled)
        tokens = printer.estimated_tokens
        _log(
            f"[vLLM] inference finished: ~{tokens} tokens, "
            f"{len(raw_text)} chars text"
        )
        if progress is not None:
            try:
                progress("postprocess", 0.98, "parsing MOSS transcript")
            except Exception:  # noqa: BLE001
                pass

        if not raw_text.strip():
            raise RuntimeError(
                "vLLM 返回为空文本;请检查 vLLM 服务端日志是否报错,"
                "或确认 --max-new-tokens 是否太小导致截断"
            )

        return _moss_text_to_result(raw_text)
