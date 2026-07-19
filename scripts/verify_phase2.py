"""Phase 2 验证脚本：跑 CLI + 各格式导出 + 复读 export 命令。"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:\Users\zhbuaa0\.conda\envs\paraformer-asr\python.exe"
AUDIO = r"C:\Users\zhbuaa0\.openclaw\media\inbound\C3142---81c7bbb6-da1e-42ce-a3e5-f07afb900f20.wav"

OUT_TRANSCRIBE = ROOT / "output" / "phase2" / "transcribe"
OUT_EXPORT = ROOT / "output" / "phase2" / "export"
OUT_TRANSCRIBE.mkdir(parents=True, exist_ok=True)
OUT_EXPORT.mkdir(parents=True, exist_ok=True)

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT)


def run(argv: list[str]) -> subprocess.CompletedProcess:
    print(f"\n$ python -m subforge.cli {' '.join(argv)}", flush=True)
    p = subprocess.run(
        [PY, "-m", "subforge.cli", *argv],
        env=ENV,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        print(f"[FAIL] exit {p.returncode}", flush=True)
        print("STDERR:", p.stderr, flush=True)
    return p


def main() -> int:
    # ---------- 1. asr models ----------
    print("=" * 70)
    print("Phase 2.4 verification")
    print("=" * 70)
    p = run(["models"])
    assert p.returncode == 0, "asr models failed"
    assert "paraformer-zh" in p.stdout, "models listing missing paraformer-zh"
    print("[OK] asr models")

    # ---------- 2. asr transcribe 全 6 格式 ----------
    formats = "srt,vtt,lrc,txt,md,json"
    p = run(["transcribe", AUDIO, "--format", formats, "-o", str(OUT_TRANSCRIBE)])
    assert p.returncode == 0, f"asr transcribe failed (rc={p.returncode})"
    print(f"[OK] asr transcribe -> {OUT_TRANSCRIBE}")

    # 文件检查
    expected = ["srt", "vtt", "lrc", "txt", "md", "json"]
    written: dict[str, Path] = {}
    for fmt in expected:
        p = OUT_TRANSCRIBE / f"C3142---81c7bbb6-da1e-42ce-a3e5-f07afb900f20.{fmt}"
        assert p.exists(), f"missing {p}"
        assert p.stat().st_size > 0, f"empty {p}"
        written[fmt] = p
    print(f"[OK] all 6 format files written, sizes: " +
          ", ".join(f"{f}={written[f].stat().st_size}B" for f in expected))

    # ---------- 3. 字段/格式 spot-check ----------

    # SRT: 第一行 = 序号；时间码 = HH:MM:SS,mmm --> HH:MM:SS,mmm
    srt = written["srt"].read_text(encoding="utf-8")
    srt_ts = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$")
    srt_lines = srt.splitlines()
    assert srt_lines[0].isdigit(), f"SRT first line not an index: {srt_lines[0]!r}"
    ts_count = sum(1 for ln in srt_lines if srt_ts.match(ln))
    print(f"[OK] SRT: {ts_count} valid timestamps, first entry:")
    print(f"     {srt_lines[0]} / {srt_lines[1]} / {srt_lines[2]}")

    # VTT: 必须以 "WEBVTT" 开头；时间码 HH:MM:SS.mmm
    vtt = written["vtt"].read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT"), f"VTT header missing: {vtt[:50]!r}"
    vtt_ts = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$")
    vtt_ts_count = sum(1 for ln in vtt.splitlines() if vtt_ts.match(ln))
    assert vtt_ts_count > 0, "no valid VTT timestamps"
    print(f"[OK] VTT: starts with WEBVTT, {vtt_ts_count} valid timestamps")

    # LRC: 时间码 [mm:ss.xx]
    lrc = written["lrc"].read_text(encoding="utf-8")
    lrc_ts = re.compile(r"^\[\d{2}:\d{2}\.\d{2}\]")
    lrc_ts_count = sum(1 for ln in lrc.splitlines() if lrc_ts.match(ln))
    print(f"[OK] LRC: {lrc_ts_count} timestamped lines")
    assert lrc_ts_count > 0, "no valid LRC timestamps"

    # TXT: 头部应有 "时长" / "段数"
    txt = written["txt"].read_text(encoding="utf-8")
    assert "时长" in txt, "TXT header missing 时长"
    assert "段数" in txt, "TXT header missing 段数"
    print(f"[OK] TXT: header contains 时长/段数")

    # MD: 表格 + ## 段落
    md = written["md"].read_text(encoding="utf-8")
    assert "| 开始 (s) | 结束 (s) |" in md, "MD table header missing"
    assert "## 完整文本" in md, "MD 完整文本 section missing"
    print(f"[OK] MD: table + 完整文本 section present")

    # JSON: 顶层 keys
    j = json.loads(written["json"].read_text(encoding="utf-8"))
    expected_keys = {"key", "text", "timestamp", "sentence_info"}
    assert expected_keys.issubset(j.keys()), f"JSON missing keys: {expected_keys - set(j.keys())}"
    print(f"[OK] JSON: top keys = {sorted(j.keys())}, segments = {len(j['sentence_info'])}")

    # ---------- 4. asr export（从已有 JSON 复读其他格式）----------
    json_path = OUT_TRANSCRIBE / "C3142---81c7bbb6-da1e-42ce-a3e5-f07afb900f20.json"
    p = run(["export", str(json_path), "--format", "srt,vtt,lrc,md", "--output-dir", str(OUT_EXPORT)])
    assert p.returncode == 0, "asr export failed"
    for fmt in ("srt", "vtt", "lrc", "md"):
        out_p = OUT_EXPORT / f"{json_path.stem}.{fmt}"
        assert out_p.exists() and out_p.stat().st_size > 0, f"missing export/{fmt}"
    print(f"[OK] asr export -> {OUT_EXPORT}")

    # 复读 SRT 与首次输出一致（同一 TranscriptResult）
    srt_via_export = (OUT_EXPORT / f"{json_path.stem}.srt").read_text(encoding="utf-8")
    assert srt_via_export == srt, "export SRT differs from transcribe SRT"
    print(f"[OK] export SRT identical to transcribe SRT")

    # ---------- 5. asr export --output（单文件）----------
    single = OUT_EXPORT / "single_via_output.srt"
    p = run(["export", str(json_path), "--format", "srt", "--output", str(single)])
    assert p.returncode == 0 and single.exists(), "single-file export failed"
    print(f"[OK] asr export --output {single.name}: {single.stat().st_size} bytes")

    # ---------- 6. legacy shim smoke（语法 + import）----------
    for shim in ("demo", "transcribe", "run_clean", "make_md"):
        shim_path = ROOT / "legacy" / f"{shim}.py"
        # 编译检查
        compile(shim_path.read_text(encoding="utf-8"), str(shim_path), "exec")
        print(f"[OK] legacy/{shim}.py compiles")

    print()
    print("=" * 70)
    print("[PASS] Phase 2.4 verification successful")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())