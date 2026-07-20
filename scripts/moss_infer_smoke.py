"""Smoke-test inference: transcribe C3142.wav with MOSS-Transcribe-Diarize."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)


AUDIO = r"C:\Users\zhbuaa0\Downloads\C3142.wav"
MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
OUT_DIR = r"F:\Program Files\openclaw\workspace\paraformer-asr\output\moss_inference"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[1/4] Loading processor ({MODEL_ID})...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, local_files_only=True
    )
    print(f"      processor loaded in {time.time()-t0:.2f}s -> {type(processor).__name__}")

    print(f"[2/4] Loading model in bf16 on cuda:0...")
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    device = resolve_device("auto")
    print(f"      model loaded in {time.time()-t1:.2f}s, "
          f"mem={torch.cuda.memory_allocated()/1e9:.2f}GB, device={device}")

    audio_info = f"{AUDIO}"
    import soundfile as sf
    info = sf.info(audio_info)
    print(f"[3/4] Audio: {info.channels}ch, {info.samplerate}Hz, "
          f"{info.frames} frames, {info.duration:.1f}s")

    MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "65536"))
    print(f"[4/4] Running inference (max_new_tokens={MAX_NEW})...")
    t2 = time.time()
    messages = build_transcription_messages(audio_info)
    result = generate_transcription(
        model,
        processor,
        messages,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        device=device,
        dtype=torch.bfloat16,
    )
    elapsed = time.time() - t2
    text = result["text"]
    truncated = result["generated_tokens"] >= MAX_NEW
    print(f"      inference took {elapsed:.1f}s, "
          f"generated_tokens={result['generated_tokens']}, "
          f"prompt_len={result['prompt_len']}, truncated={truncated}")

    raw_path = os.path.join(OUT_DIR, "C3142_moss_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"      raw transcript saved: {raw_path}")

    print()
    print("=== RAW TRANSCRIPT (first 1200 chars) ===")
    sys.stdout.buffer.write((text[:1200] + "\n").encode("utf-8"))
    sys.stdout.buffer.write(b"=== /RAW ===\n")
    sys.stdout.flush()
    print()

    segments = parse_transcript(text)
    print(f"=== PARSED SEGMENTS: {len(segments)} ===")
    for i, seg in enumerate(segments[:15]):
        print(f"  [{i:02d}] {seg.start:7.2f} - {seg.end:7.2f}  {seg.speaker}  {seg.text[:80]}")

    segs_path = os.path.join(OUT_DIR, "C3142_moss_segments.json")
    import json
    with open(segs_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text} for s in segments],
            f, ensure_ascii=False, indent=2,
        )
    print(f"segments saved: {segs_path}")

    # Compare with reference transcript in workspace (C3142_transcript.txt from subforge)
    ref_path = r"F:\Program Files\openclaw\workspace\paraformer-asr\C3142_transcript.txt"
    if os.path.exists(ref_path):
        with open(ref_path, encoding="utf-8") as f:
            ref_text = f.read()
        # MOSS output is bracketed; strip the [start][Sxx]... [end] tags
        import re
        plain = re.sub(r"\[\d+(?:\.\d+)?\]", "", text)
        plain = re.sub(r"\[S\d+\]", "", plain)
        # CER
        def cer(a: str, b: str) -> float:
            if not a:
                return float("inf") if b else 0.0
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    cost = 0 if a[i-1] == b[j-1] else 1
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
            return dp[m][n] / m
        c = cer(plain, ref_text)
        print(f"=== CER vs C3142_transcript.txt ===")
        print(f"  ref chars: {len(ref_text)}, hypothesis chars: {len(plain)}")
        print(f"  CER: {c*100:.2f}%")


if __name__ == "__main__":
    main()