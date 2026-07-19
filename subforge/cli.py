"""paraformer-asr 命令行入口。

子命令：
    asr models                                列出已注册模型
    asr transcribe AUDIO [...]                推理 + 导出
    asr export RESULT_JSON                    从已有 JSON 重新导出
    asr server [--host ...] [--port ...]      启动 HTTP+WS 服务（Phase 5）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

# Windows console 默认 cp936，强制 UTF-8 让中文 print 不乱码
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def _format_basename(audio: str, idx: int | None = None) -> str:
    """从 audio 路径/URL 派生一个可作为输出文件基名的字符串。"""
    if audio.startswith(("http://", "https://")):
        base = audio.rsplit("/", 1)[-1].split("?")[0] or "remote"
    else:
        base = Path(audio).stem
    if idx is not None:
        base = f"{idx:02d}_{base}"
    return base


# ---------- 子命令实现 ----------

def cmd_models(args: argparse.Namespace) -> int:
    from .config import load_registry

    specs, default = load_registry(args.config)
    print(f"Default model: {default}")
    print(f"Config: {Path(args.config).resolve() if args.config else '(default models.yaml)'}")
    print()
    print(f"{'NAME':<24} {'STREAM':<7} {'SPEAKERS':<9} {'PUNC':<5} {'MULTI':<6} {'MODEL ID'}")
    print("-" * 110)
    for name, spec in specs.items():
        f = spec.features
        print(
            f"{name:<24} "
            f"{'yes' if f.get('streaming') else 'no':<7} "
            f"{'yes' if f.get('speakers') else 'no':<9} "
            f"{'yes' if f.get('punc') else 'no':<5} "
            f"{'yes' if f.get('multilingual') else 'no':<6} "
            f"{spec.model}"
        )
    return 0


def _build_transcriber(model_name: str, config_path: str | None):
    from .config import load_registry
    from .models import ModelRegistry
    from .transcriber import Transcriber

    specs, default = load_registry(config_path)
    registry = ModelRegistry(specs)
    name = model_name or default
    spec = registry.spec(name)
    _log(f"loading model '{name}' -> {spec.model}")
    t0 = time.time()
    model = registry.get(name)
    _log(f"model ready in {time.time() - t0:.1f}s")
    return Transcriber(model, spec), name


def cmd_transcribe(args: argparse.Namespace) -> int:
    from . import exporters

    audios = list(args.audio)
    if not audios:
        print("[ERR] at least one audio path / URL required", file=sys.stderr)
        return 2

    transcriber, model_name = _build_transcriber(args.model, args.config)

    formats = [f.strip().lower() for f in (args.format or "").split(",") if f.strip()]

    # 检查 spk-num 是否被该模型支持
    if args.spk_num is not None and not transcriber.spec.has_spk:
        print(
            f"[ERR] model '{model_name}' has no spk_model; --spk-num 不适用",
            file=sys.stderr,
        )
        return 2

    t0 = time.time()
    results = transcriber.transcribe(
        audios if len(audios) > 1 else audios[0],
        preset_spk_num=args.spk_num,
        **({"language": args.language} if args.language else {}),
        **({"batch_size_s": args.batch_size_s} if args.batch_size_s is not None else {}),
    )
    _log(f"inference done in {time.time() - t0:.1f}s")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, r in enumerate(results):
        # 选用输入 basename 或按序号
        if len(audios) == 1:
            base = _format_basename(audios[0])
        else:
            base = _format_basename(audios[i], idx=i)

        if formats:
            for fmt in formats:
                if not out_dir:
                    print(
                        f"[ERR] --format requires --output-dir (or -o)",
                        file=sys.stderr,
                    )
                    return 2
                if fmt == "json":
                    # json 复用原始 funasr schema，单文件命名 <base>.json
                    p = exporters.export(r, "json", out_dir / f"{base}.json")
                else:
                    p = exporters.export(r, fmt, out_dir / f"{base}.{fmt}")
                _log(f"wrote {p}")
        else:
            # 默认行为：纯文本打到 stdout
            print()
            print("=" * 70)
            print(f"#{i} {audios[i] if i < len(audios) else ''}")
            print("=" * 70)
            for seg in r.segments:
                if seg.spk is not None:
                    print(f"[{seg.start:>7.2f}s - {seg.end:>7.2f}s] spk{seg.spk}: {seg.text}")
                else:
                    print(f"[{seg.start:>7.2f}s - {seg.end:>7.2f}s] {seg.text}")
            print()
            print("--- 完整文本 ---")
            print(r.text)

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """从已保存的 funasr JSON 重新导出其他格式，无需重新推理。"""
    import json
    from . import exporters
    from .transcriber import from_raw_dict

    src = Path(args.result_json)
    if not src.exists():
        print(f"[ERR] not found: {src}", file=sys.stderr)
        return 2
    raw = json.loads(src.read_text(encoding="utf-8"))
    r = from_raw_dict(raw)

    formats = [f.strip().lower() for f in (args.format or "").split(",") if f.strip()]
    if not formats:
        print("[ERR] --format required (e.g. --format srt,vtt)", file=sys.stderr)
        return 2

    if args.output:
        if len(formats) > 1:
            print(
                f"[ERR] -o only accepts a single file path; for multiple formats use --output-dir",
                file=sys.stderr,
            )
            return 2
        p = exporters.export(r, formats[0], args.output)
        _log(f"wrote {p}")
        return 0

    out_dir = Path(args.output_dir) if args.output_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = exporters.export(r, fmt, out_dir / f"{src.stem}.{fmt}")
        _log(f"wrote {p}")
    return 0


def cmd_server(args: argparse.Namespace) -> int:
    """启动 FastAPI HTTP 服务。"""
    from . import server
    # 直接转交参数
    return server.main([
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", args.log_level,
    ])


# ---------- argparse ----------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr",
        description="paraformer-asr — 多模型中文语音识别 CLI（Paraformer / SenseVoice）",
    )
    parser.add_argument(
        "--config", "-c",
        help="models.yaml 路径（默认仓库根 models.yaml）",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_models = sub.add_parser("models", help="列出已注册模型")
    p_models.set_defaults(func=cmd_models)

    p_tr = sub.add_parser("transcribe", help="推理并导出")
    p_tr.add_argument("audio", nargs="+", help="一个或多个音频路径 / URL")
    p_tr.add_argument("--model", "-m", help="模型名（默认：models.yaml 的 default）")
    p_tr.add_argument(
        "--format", "-f",
        help="输出格式（逗号分隔）：srt,vtt,lrc,txt,md,json；省略时仅打印到 stdout",
    )
    p_tr.add_argument(
        "--output-dir", "-o",
        help="输出目录；与 --format 配合时必需",
    )
    p_tr.add_argument(
        "--spk-num", type=int, default=None,
        help="强制说话人数；省略则自动检测",
    )
    p_tr.add_argument(
        "--language",
        help="语言（多语种模型用，如 auto/zh/en/yue/ja/ko）",
    )
    p_tr.add_argument(
        "--batch-size-s", type=int, default=None,
        help="动态 batch 时长阈值（秒），覆盖模型默认值",
    )
    p_tr.set_defaults(func=cmd_transcribe)

    p_ex = sub.add_parser("export", help="从已保存 JSON 重新导出其他格式")
    p_ex.add_argument("result_json", help="funasr 原始 JSON 结果文件")
    p_ex.add_argument(
        "--format", "-f", required=True,
        help="导出格式（逗号分隔）：srt,vtt,lrc,txt,md,json",
    )
    p_ex.add_argument(
        "--output", help="单文件输出路径（仅当 --format 单一格式时可用）",
    )
    p_ex.add_argument(
        "--output-dir", help="目录输出（多格式时用此）；默认与输入 JSON 同目录",
    )
    p_ex.set_defaults(func=cmd_export)

    p_sv = sub.add_parser("server", help="启动 HTTP+WebSocket 服务")
    p_sv.add_argument("--host", default="127.0.0.1")
    p_sv.add_argument("--port", type=int, default=8000)
    p_sv.add_argument("--log-level", default="info")
    p_sv.set_defaults(func=cmd_server)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())