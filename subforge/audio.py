"""音频输入辅助。

服务器 / 上传场景需要：
1. 把浏览器发来的字节流保存为临时文件（FunASR 直接吃路径）
2. 检测 ffmpeg 是否可用（webm/opus 必须）
3. 给可读但语义清晰的错误
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


# FunASR / torchaudio 不依赖 ffmpeg 就能解码的格式
_NATIVE_FORMATS = {".wav", ".flac"}


def check_ffmpeg() -> bool:
    """ffmpeg 是否在 PATH 中。"""
    return shutil.which("ffmpeg") is not None


def guess_suffix(filename: str | None, content_type: str | None) -> str:
    """根据文件名或 MIME 推断文件后缀，默认 .wav。"""
    if filename:
        suf = Path(filename).suffix.lower()
        if suf:
            return suf
    if content_type:
        ct = content_type.lower().split(";")[0].strip()
        mapping = {
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/wave": ".wav",
            "audio/flac": ".flac",
            "audio/x-flac": ".flac",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/ogg": ".ogg",
            "audio/webm": ".webm",
            "audio/opus": ".opus",
            "audio/mp4": ".mp4",
            "audio/aac": ".aac",
        }
        if ct in mapping:
            return mapping[ct]
    return ".wav"


def save_upload(content: bytes, suffix: str, *, dir: str | Path | None = None) -> Path:
    """把上传字节写到临时文件，返回路径。调用方负责删除。"""
    if dir is not None:
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
    else:
        d = Path(tempfile.gettempdir())
    fd, name = tempfile.mkstemp(suffix=suffix, prefix="paraformer_upload_", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise
    return Path(name)


def ensure_native_format(path: Path) -> Path:
    """如果后缀是 wav/flac，直接返回；否则尝试 ffmpeg 转 wav；失败时给出清晰错误。"""
    if path.suffix.lower() in _NATIVE_FORMATS:
        return path

    if not check_ffmpeg():
        raise RuntimeError(
            f"音频格式 {path.suffix} 需要 ffmpeg 才能解码；当前系统 PATH 中找不到 ffmpeg。"
            f"支持免依赖的格式：{sorted(_NATIVE_FORMATS)}。"
            f"安装 ffmpeg（conda install -c conda-forge ffmpeg）或上传 wav/flac 文件。"
        )

    out = path.with_suffix(".converted.wav")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-ac", "1", "-ar", "16000", "-f", "wav",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg 转换失败 (exit {e.returncode}); stderr: {e.stderr.decode(errors='replace')[:500]}"
        ) from e
    return out