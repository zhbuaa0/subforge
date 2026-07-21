"""Transcriber —— backend-agnostic 封装，委托给 BaseBackend 跑推理并归一化结果。

归一化要点（仍由 FunasrBackend 负责，因为 funasr 内部约定的字段名 / 毫秒时间戳
不在 MOSS 输出中存在）：
    1. `start` / `end` 统一从毫秒（funasr 内部约定）转为秒
    2. `spk` 缺失时为 None（SenseVoice / 无 spk_model 的情况）
    3. 顶层 `timestamp` 字段原样保留（如有），供高级用户使用
    4. SenseVoice 的 `<|zh|>` `<|EMO|>` 等标签通过 `rich_transcription_postprocess` 剥离

MOSS 后端的归一化在 ``subforge/backends/moss_backend.py`` 内完成（speaker 字符串
→ int、构造 ``TranscriptResult``），共用本文件的 ``Segment`` / ``TranscriptResult`` 契约。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import ModelSpec


@dataclass
class Segment:
    """一个识别片段（句/段）。

    时间单位：秒。
    spk 为 None 表示该模型不提供说话人信息（SenseVoice 或未配 spk_model）。
    """

    start: float
    end: float
    text: str
    spk: int | None = None
    # 可选：funasr 返回的逐 token 时间戳（毫秒），不导出但保留供 debug
    token_timestamps: list[list[int]] | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptResult:
    """对一份音频的识别结果。

    `text` / `segments` / `num_speakers` 是 UI/CLI/服务对外暴露的主字段；
    `language` 仅 SenseVoice 类多语种模型填；
    `raw` 保留 funasr 原始 dict 以便 debug 或后续导出（如 json）。
    """

    text: str
    segments: list[Segment] = field(default_factory=list)
    num_speakers: int | None = None
    language: str | None = None
    # 原始 funasr 输出（list[dict]，每个元素对应一份音频；多文件输入时长度 > 1）
    raw: list[dict[str, Any]] | None = None


# ---------- 工具函数 ----------

def _ms_to_s(ms: int | float) -> float:
    return float(ms) / 1000.0


def _segment_from_dict(d: dict[str, Any]) -> Segment:
    spk_raw = d.get("spk")
    return Segment(
        start=_ms_to_s(d.get("start", 0)),
        end=_ms_to_s(d.get("end", 0)),
        text=str(d.get("text", "") or ""),
        spk=int(spk_raw) if spk_raw is not None else None,
        token_timestamps=d.get("timestamp"),
    )


def _normalize(raw_results: Sequence[dict[str, Any]]) -> list[TranscriptResult]:
    """把 funasr `generate()` 返回的 list[dict] 归一化为 list[TranscriptResult]。"""
    out: list[TranscriptResult] = []
    for res in raw_results:
        if not isinstance(res, dict):
            continue
        segs: list[Segment] = []
        for s in res.get("sentence_info") or []:
            if isinstance(s, dict):
                segs.append(_segment_from_dict(s))

        spks = [s.spk for s in segs if s.spk is not None]
        num_speakers = (max(spks) + 1) if spks else None

        # SenseVoice / 多语种可能把 language 放在顶层或 sentence_info 内
        lang = res.get("language") or res.get("lang")
        if not lang and segs:
            # 部分版本在 sentence_info[0].text 里塞了 <|zh|> 标签；postprocess 会处理
            pass

        out.append(
            TranscriptResult(
                text=str(res.get("text", "") or ""),
                segments=segs,
                num_speakers=num_speakers,
                language=lang,
                raw=[res] if res is not None else None,
            )
        )
    return out


def _apply_postprocess(results: list[TranscriptResult], spec: ModelSpec) -> None:
    """按 spec.postprocess 名字分派后处理。原地修改。"""
    name = spec.postprocess
    if not name:
        return
    if name == "rich_transcription":
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "postprocess=rich_transcription 需要 funasr >= 1.x 但当前 funasr 不可用或 API 变更"
            ) from e
        for r in results:
            # SenseVoice 把 <|lang|> <|EMO|> <|Event_UNK|> 等标签拼在 raw["text"] 里
            raw_text = (r.raw or [{}])[0].get("text", "") if r.raw else ""
            if not raw_text:
                continue
            cleaned = rich_transcription_postprocess(raw_text)
            r.text = cleaned
            # SenseVoice 没 sentence_info 时 segments 是空的；若有（如配了 vad_model），把全文塞到第一个 segment
            if r.segments:
                r.segments[0].text = cleaned
    else:
        raise ValueError(f"未知 postprocess: {name!r}")


# ---------- 主类 ----------

class Transcriber:
    """围绕一个已加载的 BaseBackend + ModelSpec 的封装（backend-agnostic）。

    一般不需要直接构造：见 `models.ModelRegistry` 与 `cli.transcribe`。
    """

    def __init__(self, backend: Any, spec: ModelSpec):
        self.backend = backend
        self.spec = spec

    def transcribe(
        self,
        audio: str | Iterable[str],
        *,
        preset_spk_num: int | None = None,
        progress: "Callable[[str, float, str], None] | None" = None,
        **generate_overrides: Any,
    ) -> list[TranscriptResult]:
        """对单个或多个音频做识别（委托给 backend）。

        Args:
            audio: 单个路径/URL 或可迭代的多个；backend 实现自行决定如何处理。
            preset_spk_num: 强制说话人数；仅 FunasrBackend 使用，MossBackend 忽略。
            progress: 可选回调 ``(phase, pct, msg)``；用于在 Web UI SSE 上报进度。
            **generate_overrides: 临时覆盖 spec.generate 中的字段。

        Returns:
            与输入一一对应的 TranscriptResult 列表。
        """
        # FunasrBackend 接受 ``preset_spk_num`` 关键字；MossBackend 忽略它
        kwargs: dict[str, Any] = dict(generate_overrides)
        if progress is not None:
            kwargs["progress"] = progress
        if preset_spk_num is not None:
            return self.backend.transcribe(
                audio, preset_spk_num=preset_spk_num, **kwargs
            )
        return self.backend.transcribe(audio, **kwargs)


# ---------- 工具：从保存的 funasr JSON 还原 TranscriptResult ----------

def from_raw_dict(raw: dict[str, Any]) -> TranscriptResult:
    """把 funasr 原始输出 dict 还原成 TranscriptResult。

    用于 `asr export` 从磁盘 JSON 重新生成字幕，不需要重跑推理。

    注意：start/end 单位 funasr 内部是毫秒，这里统一转秒。
    """
    if not isinstance(raw, dict):
        raise TypeError(f"expected dict, got {type(raw).__name__}")
    results = _normalize([raw])
    return results[0] if results else TranscriptResult(text="")