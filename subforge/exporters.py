"""把 TranscriptResult 序列化为常见格式。

设计原则：
- 纯标准库（不依赖 funasr / numpy / 任何第三方包），方便在服务端、CI 单独使用
- 每个 to_xxx 函数返回 str，export() 负责落盘
- 说话人标签在 spk=None 时自动省略（SenseVoice / 单说话人 等场景）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .transcriber import Segment, TranscriptResult


# ---------- 时间码格式化 ----------

def _srt_ts(seconds: float) -> str:
    """SRT 时间码：HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600 * 1000)
    m, rem = divmod(rem, 60 * 1000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_ts(seconds: float) -> str:
    """WebVTT 时间码：HH:MM:SS.mmm"""
    return _srt_ts(seconds).replace(",", ".")


def _lrc_ts(seconds: float) -> str:
    """LRC 时间码：[mm:ss.xx]（百毫秒精度，无小时位）"""
    if seconds < 0:
        seconds = 0.0
    total_cs = int(round(seconds * 100))  # centiseconds
    m, cs = divmod(total_cs, 60 * 100)
    s, cc = divmod(cs, 100)
    return f"{m:02d}:{s:02d}.{cc:02d}"


# ---------- 说话人标签辅助 ----------

def _spk_label(seg: Segment, prefix: str = "Speaker ", numbered: bool = True) -> str | None:
    """返回 `Speaker 1:` 形式的标签；spk 缺失时返回 None。

    numbered=True（默认）：Speaker 1, Speaker 2 ...
    numbered=False：        Speaker A, Speaker B ... （适合对外展示）
    """
    if seg.spk is None:
        return None
    if numbered:
        return f"{prefix}{seg.spk + 1}"
    # A, B, C, ... Z, AA, AB ...
    n = seg.spk
    letters = ""
    while True:
        letters = chr(ord("A") + (n % 26)) + letters
        n = n // 26 - 1
        if n < 0:
            break
    return f"{prefix}{letters}"


# ---------- 格式实现 ----------

def to_srt(r: TranscriptResult, *, speaker_labels: bool = True) -> str:
    """导出 SRT 字幕。

    SRT 没有官方的说话人标签规范，约定：在 text 前面加 `Speaker N:` 前缀。
    speaker_labels=False 时去掉前缀。
    """
    out: list[str] = []
    for i, seg in enumerate(r.segments, start=1):
        start, end = _srt_ts(seg.start), _srt_ts(seg.end)
        if end == start:
            # 防 0 时长导致播放器跳过
            end = _srt_ts(seg.end + 0.001)
        text = seg.text.replace("\n", " ").strip()
        if speaker_labels:
            label = _spk_label(seg)
            if label is not None:
                text = f"{label}: {text}"
        out.append(str(i))
        out.append(f"{start} --> {end}")
        out.append(text)
        out.append("")
    return "\n".join(out)


def to_vtt(r: TranscriptResult, *, speaker_labels: bool = True) -> str:
    """导出 WebVTT 字幕，用原生 `<v Speaker N>...</v>` 语音标签。"""
    lines = ["WEBVTT", ""]
    for seg in r.segments:
        start, end = _vtt_ts(seg.start), _vtt_ts(seg.end)
        if end == start:
            end = _vtt_ts(seg.end + 0.001)
        text = seg.text.replace("\n", " ").strip()
        if speaker_labels and (label := _spk_label(seg)):
            text = f"<v {label}>{text}</v>"
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_lrc(r: TranscriptResult, *, speaker_labels: bool = True) -> str:
    """导出 LRC 歌词格式（[mm:ss.xx] 前缀）。

    无小时位。文本较长的段会被截断成多行（每行 ≤ 80 字）。
    """
    out: list[str] = []
    for seg in r.segments:
        ts = f"[{_lrc_ts(seg.start)}]"
        text = seg.text.strip().replace("\n", " ")
        label = _spk_label(seg) if speaker_labels else None
        # 按 80 字切行（中文按字符计；粗略）
        max_w = 80 - (len(label) + 2 if label else 0) - len(ts) - 3
        if max_w < 10:
            max_w = 10
        chunks = [text[i:i + max_w] for i in range(0, len(text), max_w)] or [""]
        for j, chunk in enumerate(chunks):
            prefix = ts  # 标准 LRC：每行都必须有 [mm:ss.xx]
            line = f"{prefix}{chunk}"
            if j == 0 and label:
                line = f"{prefix} {label}: {chunk}"
            out.append(line)
    return "\n".join(out) + "\n"


def to_txt(r: TranscriptResult, *, speaker_labels: bool = True, header: bool = True) -> str:
    """人类可读 txt。

    头部（可选）：模型名 / 说话人数 / 时长。
    主体：每段一行 `[ start - end] spkN: text`。
    末尾：完整拼接文本。
    """
    lines: list[str] = []
    if header:
        if r.num_speakers is not None:
            spk_info = f"说话人: {r.num_speakers}"
        else:
            spk_info = "说话人: 未知"
        duration = max((s.end for s in r.segments), default=0.0)
        lines.append("=" * 70)
        lines.append("转录结果")
        lines.append("=" * 70)
        lines.append(f"时长: {duration:.2f}s  {spk_info}  段数: {len(r.segments)}")
        lines.append("")
        lines.append("【分段识别结果】")
        for seg in r.segments:
            label = _spk_label(seg) if speaker_labels else None
            if label is not None:
                lines.append(f"[{seg.start:>7.2f}s - {seg.end:>7.2f}s] {label}: {seg.text}")
            else:
                lines.append(f"[{seg.start:>7.2f}s - {seg.end:>7.2f}s] {seg.text}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("【完整文本 (合并)】")
        lines.append("=" * 70)
    lines.append(r.text)
    return "\n".join(lines)


def to_md(r: TranscriptResult, *, header: bool = True) -> str:
    """Markdown 表格（含完整文本段落）。"""
    lines: list[str] = []
    if header:
        lines.append("# 转录结果\n")
        if r.num_speakers is not None:
            lines.append(f"**说话人数:** {r.num_speakers}  ")
        duration = max((s.end for s in r.segments), default=0.0)
        lines.append(f"**时长:** {duration:.2f}s  ")
        lines.append(f"**段数:** {len(r.segments)}\n")
        lines.append("## 分段识别结果\n")
    lines.append("| 开始 (s) | 结束 (s) | 说话人 | 内容 |")
    lines.append("|---:|---:|:---:|:---|")
    for seg in r.segments:
        text = (seg.text or "").replace("|", "\\|").replace("\n", " ")
        label = _spk_label(seg) or "—"
        lines.append(f"| {seg.start:.2f} | {seg.end:.2f} | {label} | {text} |")
    if header:
        lines.append("\n## 完整文本\n")
        lines.append(r.text)
    return "\n".join(lines) + "\n"


def to_json(r: TranscriptResult) -> str:
    """导出 funasr 原始 JSON 形状（与 run_clean.py 输出兼容）。

    不暴露 TranscriptResult 内部结构，方便下游脚本继续按 funasr schema 解析。
    """
    if r.raw and isinstance(r.raw, list) and r.raw:
        payload: Any = r.raw[0] if len(r.raw) == 1 else r.raw
    else:
        # 没 raw 时回退到 TranscriptResult 序列化
        payload = {
            "text": r.text,
            "num_speakers": r.num_speakers,
            "language": r.language,
            "sentence_info": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "spk": s.spk,
                }
                for s in r.segments
            ],
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------- 调度 ----------

_FORMATTERS = {
    "srt": to_srt,
    "vtt": to_vtt,
    "lrc": to_lrc,
    "txt": to_txt,
    "md": to_md,
    "json": to_json,
}


def available_formats() -> list[str]:
    return list(_FORMATTERS.keys())


def render(r: TranscriptResult, fmt: str, **kw: Any) -> str:
    """公共调度：fmt -> 字符串。

    与 `export(r, fmt, path)` 同源，但只返回字符串不写盘，方便嵌入 JSON 响应。
    """
    fmt = fmt.lower()
    if fmt not in _FORMATTERS:
        raise ValueError(f"不支持的格式 '{fmt}'；可选：{available_formats()}")
    return _FORMATTERS[fmt](r, **kw)


def export(r: TranscriptResult, fmt: str, path: str | Path, **kw: Any) -> Path:
    """把 TranscriptResult 写成 `fmt` 格式到 `path`。返回写入路径。"""
    fmt = fmt.lower()
    if fmt not in _FORMATTERS:
        raise ValueError(f"不支持的格式 '{fmt}'；可选：{available_formats()}")
    content = _FORMATTERS[fmt](r, **kw)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p