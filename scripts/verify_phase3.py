"""Phase 3 验证：SenseVoice 模型加载 + rich_transcription 后处理剥离 <|...|> 标签。"""
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

OUT = ROOT / "output" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT)


def run(argv: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    print(f"\n$ python -m subforge.cli {' '.join(argv)}", flush=True)
    p = subprocess.run(
        [PY, "-m", "subforge.cli", *argv],
        env=ENV,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if p.returncode != 0:
        print(f"[FAIL] exit {p.returncode}", flush=True)
        print("STDERR (tail):", p.stderr[-1500:], flush=True)
    return p


def main() -> int:
    print("=" * 70)
    print("Phase 3.2 verification — SenseVoice + rich_transcription")
    print("=" * 70)

    # ---------- 1. asr models 列出 sensevoice ----------
    p = run(["models"])
    assert p.returncode == 0, "asr models failed"
    assert "sensevoice" in p.stdout, "sensevoice missing from models list"
    print("[OK] sensevoice registered")

    # ---------- 2. 用 sensevoice 转写 ----------
    p = run(["transcribe", AUDIO, "--model", "sensevoice", "--format", "json,txt", "-o", str(OUT)],
            timeout=600)
    assert p.returncode == 0, "sensevoice transcribe failed"
    print(f"[OK] sensevoice transcribe -> {OUT}")

    # 找到输出 JSON
    json_candidates = list(OUT.glob("*.json"))
    assert json_candidates, f"no json output in {OUT}"
    sensevoice_json = json_candidates[0]

    raw = json.loads(sensevoice_json.read_text(encoding="utf-8"))
    print(f"[INFO] sensevoice raw keys: {sorted(raw.keys())}")
    print(f"[INFO] sensevoice raw text (first 200 chars): {raw.get('text', '')[:200]!r}")

    # ---------- 3. raw 文本应含 <|zh|> <|EMO|> 之类的标签 ----------
    raw_text = raw.get("text", "")
    assert "<|zh|>" in raw_text or "<|en|>" in raw_text or "<|yue|>" in raw_text, \
        f"raw text missing language tag: {raw_text[:200]!r}"
    print(f"[OK] raw text contains language tag (e.g. <|zh|>)")

    # ---------- 4. 通过 transcriber 归一化后，标签应被剥离 ----------
    # 用 Python 直接验证：sensevoice 的 text 经过 rich_transcription_postprocess 应不含 <|...|>
    # 这一步通过 transcriber.transcribe() 已经做过（CLI 调用时），验证 txt 输出无标签
    txt_candidates = list(OUT.glob("*.txt"))
    assert txt_candidates, f"no txt output in {OUT}"
    sensevoice_txt = txt_candidates[0].read_text(encoding="utf-8")

    tag_pattern = re.compile(r"<\|[^|]+?\|>")
    tags_in_txt = tag_pattern.findall(sensevoice_txt)
    assert not tags_in_txt, f"未剥离的标签残留在 txt 中: {set(tags_in_txt)}"
    print(f"[OK] no <|...|> tags in exported txt")

    # 顶层 txt 应该展示处理后的 text（不是 raw）
    assert "时长" in sensevoice_txt, "txt 头部缺失"
    assert "段数" in sensevoice_txt, "txt 头部缺失 段数"
    # 没有 sentence_info 时 num_speakers 应为 None，header 应展示 "说话人: 未知"
    assert "说话人: 未知" in sensevoice_txt, \
        f"sensevoice 应展示 '说话人: 未知'，实际 txt 头: {sensevoice_txt[:200]!r}"
    print(f"[OK] txt header correctly shows '说话人: 未知'")

    # ---------- 5. feature-gating: --spk-num 应被 sensevoice 拒绝 ----------
    p = run(["transcribe", AUDIO, "--model", "sensevoice", "--spk-num", "2"])
    assert p.returncode != 0, "sensevoice 接受了 --spk-num（不应该）"
    assert "no spk_model" in p.stderr or "spk" in p.stderr.lower(), \
        f"expected feature-gating error, got: {p.stderr[-300:]}"
    print(f"[OK] --spk-num 被 sensevoice 拒绝（feature-gating 生效）")

    # ---------- 6. 输出文本非空且为合理中文 ----------
    # 取 txt 的 "完整文本" 段
    m = re.search(r"完整文本[\s\S]*?\n=+\n([\s\S]+)$", sensevoice_txt)
    if m:
        full_text = m.group(1).strip()
    else:
        # 退化：取整文件
        full_text = sensevoice_txt
    assert len(full_text) > 20, f"full text too short: {len(full_text)}"
    assert re.search(r"[一-鿿]", full_text), "no Chinese characters in output"
    print(f"[OK] sensevoice produced {len(full_text)} chars of Chinese text")
    print(f"     preview: {full_text[:100]!r}")

    print()
    print("=" * 70)
    print("[PASS] Phase 3.2 verification successful")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())