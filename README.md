# subforge

> **Subtitle forge from Chinese ASR — now with dual backends: FunASR + MOSS.**
> Turn audio into editable subtitles — drop SRT / VTT / LRC straight into Premiere, Final Cut, CapCut, JianYing, Audacity.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pip-install-blue.svg)](https://pypi.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![FunASR](https://img.shields.io/badge/powered%20by-FunASR-orange.svg)](https://github.com/modelscope/FunASR)
[![MOSS](https://img.shields.io/badge/powered%20by-MOSS--Transcribe--Diarize-purple.svg)](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)

## What is subforge?

`subforge` is a subtitle-first Chinese ASR toolkit for **video editors, podcast producers, and content creators**. It supports two independent backends:

- **FunASR backend** (default) — wraps ModelScope's FunASR (Paraformer-large + VAD + punctuation + CAM++ speaker diarization)
- **MOSS backend** — wraps [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), a 0.9B end-to-end transformer model that jointly transcribes and diarizes in a single forward pass

Both backends share the same CLI flags, REST API, subtitle exporters, and web UI. Switch between them with `--model`.

Features:

- **Multi-format subtitle export** — SRT / WebVTT / LRC, ready for import into any NLE
- **Multi-speaker diarization** — auto-detects who said what (great for interviews / podcasts / vlogs)
- **Built-in web UI** — drag-and-drop upload, see results, download all formats in one click
- **REST API + WebSocket** — drop into your existing pipeline or video editor plugin
- **Multiple backends** — Paraformer-large with VAD/punc/spk pipeline, SenseVoice (multilingual, lightweight), streaming variant, *and* MOSS-Transcribe-Diarize (e2e, long-form meetings)
- **Pure-Python exporters** — zero external dependencies for the subtitle rendering itself

## Why?

Because clipping/transcribing by hand takes hours. subforge turns a 5-minute Chinese podcast into a 190-segment, 2-speaker, time-coded, punctuated transcript in ~15 seconds on an RTX 4070 Ti.

## Quick start

```bash
# 1. Install (requires conda env with torch+cu121 + funasr or transformers already set up)
pip install -e .

# 2. List registered models
asr models

# 3a. Transcribe with FunASR (default)
asr transcribe interview.wav \
    --model paraformer-zh \
    --format srt,vtt,lrc,txt,md,json \
    -o output/

# 3b. Transcribe with MOSS (joint ASR + diarization, good for long meetings)
asr transcribe meeting.wav \
    --model moss-transcribe-diarize \
    --max-new-tokens 65536 \
    --format srt,json,txt \
    -o output/

# 4. (optional) Start the web UI
asr server --host 0.0.0.0 --port 8000
#  → open http://localhost:8000/
```

## Features in detail

### 1. Subtitle export — six formats, zero external deps

| Format | Use case | Imports into |
|---|---|---|
| **SRT** | Standard subtitle; universal | Premiere, Final Cut, DaVinci, CapCut, JianYing, Aegisub |
| **WebVTT** | Web video (`<track>` element) | HTML5 `<video>`, YouTube captions, OBS |
| **LRC** | Lyric format with timestamps | Music players, karaoke |
| **TXT** | Plain reading text with timestamps | Any editor |
| **MD** | Markdown table for review | GitHub PRs, Notion |
| **JSON** | FunASR raw schema for re-export | Programmatic |

Speaker labels are first-class: SRT prefixes `Speaker 1:` and VTT uses native `<v Speaker 1>` voice tags.

### 2. Multi-speaker diarization — two approaches

subforge supports two fundamentally different diarization strategies, selected automatically by which model you use:

| Approach | Models | How it works |
|---|---|---|
| **Pipeline** (FunASR) | `paraformer-zh`, `seaco-paraformer-zh` | VAD → ASR → CAM++ embeddings → spectral clustering + eigengap K |
| **End-to-end** (MOSS) | `moss-transcribe-diarize` | Single transformer generates `[start][Sxx]...text...[end]` directly — joint ASR + diarization in one pass |

```bash
# Pipeline: let the engine decide how many speakers
asr transcribe interview.wav --model paraformer-zh --format srt,json

# Pipeline: force a specific count (e.g., "I know it's 3 people")
asr transcribe interview.wav --model paraformer-zh --spk-num 3

# End-to-end: MOSS auto-detects speakers (no --spk-num flag)
asr transcribe meeting.wav --model moss-transcribe-diarize --format srt,json
```

Under the hood (FunASR pipeline): VAD → Paraformer (text + per-token timestamps) → CAM++ (192-dim speaker embedding per segment) → spectral clustering with eigengap for K → label back-fill.

### 3. Web UI — no command line needed

`asr server` ships with a single-page HTML UI:

- Drag-and-drop upload
- Model dropdown (auto-populated from `/models`)
- Format checkboxes
- Speaker count override
- Live stats: model, speaker count, segment count, text length
- Per-format download buttons

Open `http://<your-host>:8000/` after starting the server.

### 4. REST API — programmatic access

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Transcribe (multipart upload, FunASR)
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@interview.wav" \
  -F "model=paraformer-zh" \
  -F "formats=srt,vtt"

# Transcribe (multipart upload, MOSS)
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@meeting.wav" \
  -F "model=moss-transcribe-diarize" \
  -F "formats=srt,json"
```

Returns JSON with `text`, `num_speakers`, `segments[]`, and (if requested) inline SRT/VTT/etc.

### 5. Calling from Python code — server client

```python
import httpx
from pathlib import Path

# ── List models ──
r = httpx.get("http://localhost:8000/models")
print(r.json()["default"])               # "paraformer-zh"

# ── Transcribe with FunASR (fast) ──
with open("interview.wav", "rb") as f:
    r = httpx.post(
        "http://localhost:8000/transcribe",
        data={"model": "paraformer-zh", "formats": "srt,json"},
        files={"audio": f},
    )
data = r.json()
print(data["num_speakers"])              # 2
print(data["segments"][0]["text"])       # "哦 看这个"
Path("out.srt").write_text(data["srt"], encoding="utf-8")

# ── Transcribe with MOSS (e2e, longer) ──
with open("meeting.wav", "rb") as f:
    r = httpx.post(
        "http://localhost:8000/transcribe",
        data={"model": "moss-transcribe-diarize", "formats": "srt,json"},
        files={"audio": f},
    )
data = r.json()
print(f"{data['num_speakers']} speakers, {len(data['segments'])} segments")
```

### 6. WebSocket streaming — live / real-time

For live transcription (interview capture, podcast recording, live captioning):

```python
import websockets, json, asyncio

async def main():
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        await ws.send(json.dumps({"model": "paraformer-zh-streaming"}))
        ready = json.loads(await ws.recv())
        print(ready)  # {"status": "ready", ...}

        # Send wav bytes; server replies with partial / final
        with open("interview.wav", "rb") as f:
            await ws.send(f.read())
        msg = json.loads(await ws.recv())
        print(msg["partial"], msg["text"])

        await ws.send(json.dumps({"action": "final"}))
        print(json.loads(await ws.recv())["text"])

asyncio.run(main())
```

## CLI reference

```
asr models                              # list registered models (shows BACKEND column)
asr transcribe <audio>...               # run inference + export
    --model NAME                        # default: models.yaml's `default`
    --format srt,vtt,txt,md,lrc,json    # comma-separated; omit for stdout
    -o DIR                              # output dir (required if --format)
    --spk-num N                         # force speaker count (FunASR only; MOSS ignores)
    --language auto|zh|en|yue|ja|ko     # multilingual models
    --batch-size-s N                    # override model's default (FunASR)
    --max-new-tokens N                  # MOSS generation limit (default: 65536)
asr export <result.json>                # re-export from saved JSON (no re-inference)
    --format srt,vtt,...
    -o FILE                             # single file
    --output-dir DIR                    # multiple formats
asr polish <result.json>                # AI polish via LLM (works on any backend)
    --provider minimax|deepseek|openai
    --format srt,vtt,txt,md
asr server [--host 0.0.0.0] [--port 8000]
```

## Python API

```python
from subforge import load_registry, ModelRegistry, Transcriber

specs, default = load_registry()
registry = ModelRegistry(specs)

# FunASR (pipeline backend)
tr = Transcriber(registry.get("paraformer-zh"), specs["paraformer-zh"])
results = tr.transcribe("interview.wav")
r = results[0]
print(r.text)                # full transcript
print(r.num_speakers)        # auto-detected count
print(len(r.segments))       # 190 for a 5-min interview

# MOSS (e2e backend)
tr2 = Transcriber(registry.get("moss-transcribe-diarize"), specs["moss-transcribe-diarize"])
r2 = tr2.transcribe("meeting.wav", max_new_tokens=65536)[0]
print(f"{r2.num_speakers} speakers, {len(r2.segments)} segments")

# Save subtitles (same API regardless of backend)
from subforge import exporters
exporters.export(r2, "srt", "out.srt")
exporters.export(r2, "vtt", "out.vtt", speaker_labels=True)
```

## Registered models

| Name | Backend | Model ID | Languages | Speakers | Streaming | Best for |
|---|---|---|---|---|---|---|
| `paraformer-zh` (default) | FunASR | `iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn` | Chinese | auto | no | General ASR |
| `seaco-paraformer-zh` | FunASR | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | Chinese | no* | no | Overlapped speech |
| `sensevoice` | FunASR | `iic/SenseVoiceSmall` | zh/en/yue/ja/ko | no | no | Multilingual |
| `paraformer-zh-streaming` | FunASR | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | Chinese | no | yes | Live captioning |
| `moss-transcribe-diarize` | **MOSS** | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | 50+ languages | auto (e2e) | no | Long meetings, podcasts |

\* SeACo handles multi-speaker internally; speaker labels may not be reliable.

Add your own in [`models.yaml`](models.yaml). Set `backend: moss` to use the MOSS adapter, or omit it (defaults to `funasr`).

## How multi-speaker recognition works

The FunASR pipeline uses a 5-step pipeline coordinated by FunASR. MOSS-Transcribe-Diarize works differently — it generates `[start][Sxx]...text...[end]` natively in a single transformer forward pass (see [end-to-end section](#moss-end-to-end-diarization)).

```
audio.wav
  │
  ▼
[1] VAD (FSMN) → segments of "speech present"
  │
  ▼
[2] FunASR ASR → per-segment text + per-token timestamps
  │
  ▼
[3] CAM++ → 192-dim speaker embedding per segment
  │
  ▼
[4] Spectral clustering + eigengap → choose K, assign speaker labels
  │
  ▼
[5] Back-fill: sentence_info[{start, end, text, spk, timestamp}]
```

When you pass `--spk-num N`, you override step 4 to use a fixed K instead of auto-detection. Short audios (<20 segments) often default to single-speaker; use `--spk-num` to force.

### MOSS end-to-end diarization

[MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) is a 0.9B transformer that jointly transcribes and diarizes in **a single forward pass**. Instead of the 5-step pipeline above, it:

- Encodes the entire audio at once via a Whisper-style encoder
- Generates `[start<timestamp>][S<speaker>]transcribed text[end<timestamp>]` tokens autoregressively
- Supports 50+ languages (Whisper codebook), no separate VAD or speaker embedding model needed

**When to use MOSS:**

| Scenario | Recommend | Reason |
|---|---|---|
| Long-form meetings (15–60 min) | MOSS | No VAD chunk limits; single-pass diarization |
| Podcasts with 2–4 speakers | MOSS | Cleaner speaker separation than pipeline |
| Overlapped speech / cross-talk | MOSS | E2E model handles overlap better than clustering |
| Fast transcription (<10 min clips) | FunASR | ~15s vs ~5 min for MOSS on RTX 4070 Ti |
| Streaming / real-time | FunASR | MOSS lacks streaming support |

**Limitations:**
- MOSS generates ~3000 tokens for 5 min of audio on a 12 GB GPU; very long sessions (>60 min) may hit `max_new_tokens` limits
- No `--spk-num` override — speaker count is auto-detected
- ~5–6 min inference on 5 min of audio (RTX 4070 Ti) vs ~15s for FunASR pipeline
- Requires HuggingFace `transformers`; weights (~1.8 GB) are downloaded on first use

## Project structure

```
subforge/
├── pyproject.toml          # PEP 621, installable via pip
├── models.yaml             # model registry (edit me to add models)
├── subforge/               # Python package source
│   ├── __init__.py
│   ├── backends/            # ASR model backends
│   │   ├── __init__.py
│   │   ├── base.py          # BaseBackend protocol
│   │   ├── funasr_backend.py # FunASR AutoModel adapter (default)
│   │   └── moss_backend.py  # MOSS-Transcribe-Diarize adapter
│   ├── config.py           # load_registry() / ModelSpec
│   ├── models.py           # ModelRegistry (dispatches to backends)
│   ├── transcriber.py      # Transcriber (delegates to backend) + TranscriptResult normalization
│   ├── exporters.py        # pure-stdlib subtitle exporters
│   ├── audio.py            # upload + ffmpeg fallback
│   ├── streaming.py        # WebSocket StreamingSession
│   ├── server.py           # FastAPI app + lifespan + /health /models /transcribe /ws/stream
│   ├── cli.py              # argparse subcommands
│   └── web/
│       └── index.html      # single-page Web UI
├── legacy/                 # backward-compat shims for old scripts
├── scripts/                # verification scripts (verify_phase1..6.py)
├── output/                 # sample outputs
├── README.md               # English (this file)
├── README.zh.md            # 中文版本
├── EXTENSION_PLAN.md       # development roadmap / changelog
└── LICENSE
```

## Caveats & known limits

- **`ffmpeg`** — needed for non-wav/flac uploads over HTTP. Local CLI works with `.wav`/`.flac` out of the box. Install via `conda install -c conda-forge ffmpeg`.
- **Streaming model RTF** — `paraformer-zh-streaming` is ~30× slower than batch on long audio (RTF ~1.0 on short clips, ~120× on the full 5-min file). The infrastructure is there; quality / speed trade-off is a FunASR model property.
- **MOSS backend inference speed** — ~5–6 min on 5 min of audio (RTX 4070 Ti). The model runs at ~RTF 1.0 on GPU. For fast turnaround on short clips, use `paraformer-zh` (~15s on the same 5 min). MOSS shines on long-form meetings where its single-pass diarization simplifies the pipeline.
- **MOSS first-run download** — ~1.8 GB of safetensors are downloaded from HuggingFace Hub on first use. Cached in `~/.cache/huggingface/hub/`.
- **MOSS `--spk-num` not supported** — speaker count is auto-detected by the model. Use `--max-new-tokens` to control generation length (default 65536).
- **No CORS** — the server doesn't set CORS headers. Cross-origin browser clients need a CORS proxy, or deploy the UI on the same origin.
- **Windows console encoding** — Windows console is GBK by default; CLI auto-reconfigures to UTF-8. If you see garbled Chinese, set `PYTHONIOENCODING=utf-8`.

## Requirements

- Python 3.10+
- PyTorch with CUDA support (tested on 2.5.1+cu121)
- **FunASR backend** requires `funasr >= 1.3` and `modelscope >= 1.38` (conda install recommended)
- **MOSS backend** requires `transformers >= 5.0` and `moss-transcribe-diarize >= 0.1.0` (installed via `pip install subforge[moss-runtime]`; torch stays conda-managed)
- `fastapi`, `uvicorn[standard]`, `python-multipart`, `websockets`, `pyyaml` (auto-installed by `pip install -e .`)
- ffmpeg (only for non-wav HTTP uploads)
- GPU: ≥ 8 GB VRAM recommended (RTX 4070 Ti 12 GB verified)

## License

MIT

## Acknowledgments

- [FunASR](https://github.com/modelscope/FunASR) — the underlying ASR / VAD / punctuation / speaker toolkit
- [ModelScope](https://www.modelscope.cn/) — model hosting for FunASR models
- [Paraformer](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn) — the workhorse ASR model
- [CAM++](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) — speaker embedding model
- [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) — end-to-end transcription + diarization model (0.9B transformer)