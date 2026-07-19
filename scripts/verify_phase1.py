"""Phase 1.2 验证脚本：跑 C3142.wav，对照 C3142_result.json 字段一致性。"""
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout (Windows console default is GBK)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from subforge.config import load_registry
from subforge.models import ModelRegistry
from subforge.transcriber import Transcriber

ROOT = Path(__file__).resolve().parent.parent
AUDIO = r"C:\Users\zhbuaa0\.openclaw\media\inbound\C3142---81c7bbb6-da1e-42ce-a3e5-f07afb900f20.wav"
OUT_DIR = ROOT / "output" / "phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    specs, default = load_registry()
    registry = ModelRegistry(specs)

    print(f"[INFO] building model {default}...", flush=True)
    t0 = time.time()
    model = registry.get(default)
    print(f"[INFO] model built in {time.time()-t0:.1f}s", flush=True)

    tr = Transcriber(model, specs[default])
    print(f"[INFO] transcribing {AUDIO}", flush=True)
    t0 = time.time()
    results = tr.transcribe(AUDIO)
    print(f"[INFO] inference done in {time.time()-t0:.1f}s", flush=True)

    assert len(results) == 1, f"expected 1 result, got {len(results)}"
    r = results[0]

    # ---- 字段对照 ----
    old = json.loads((ROOT / "C3142_result.json").read_text(encoding="utf-8"))
    old_segs = old.get("sentence_info", [])
    print()
    print("=" * 70)
    print("Phase 1.2 verification")
    print("=" * 70)
    print(f"OLD (C3142_result.json):  segments={len(old_segs)}, text_len={len(old.get('text',''))}, spks={sorted(set(s.get('spk') for s in old_segs))}")
    print(f"NEW (transcriber):        segments={len(r.segments)}, text_len={len(r.text)}, num_speakers={r.num_speakers}, spks={sorted(set(s.spk for s in r.segments))}")

    # 关键字段一致性
    checks = []
    checks.append(("segment_count_match", len(r.segments) == len(old_segs)))
    checks.append(("num_speakers_match", r.num_speakers == max((s.get('spk',0) for s in old_segs), default=0) + 1))
    # text 不一定 byte-by-byte 相等（浮点抖动 / tokenizer 微差），允许 ±5 字符
    text_diff = abs(len(r.text) - len(old.get('text','')))
    checks.append(("text_len_close", text_diff <= 5))
    # 头两条 segment 的 spk/start/end 应该接近
    if r.segments and old_segs:
        s_new, s_old = r.segments[0], old_segs[0]
        checks.append(("first_spk_match", s_new.spk == s_old.get('spk')))
        checks.append(("first_start_close_s", abs(s_new.start - s_old.get('start',0)/1000) < 0.1))

    print()
    print("Checks:")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    # 把原始 funasr 输出也写出来方便 diff
    new_json = OUT_DIR / "C3142_result_new.json"
    new_json.write_text(json.dumps(r.raw[0], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {new_json}")

    # 顶层字段对照
    print(f"\nOLD top keys: {sorted(old.keys())}")
    print(f"NEW top keys: {sorted(r.raw[0].keys())}")

    if not all(ok for _, ok in checks):
        print("\n[FAIL] not all checks passed")
        return 1
    print("\n[PASS] Phase 1.2 verification successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())