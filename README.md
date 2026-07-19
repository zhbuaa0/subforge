# subforge

> **Subtitle forge from Chinese ASR.**
> Turn audio into editable subtitles — drop SRT / VTT / LRC straight into Premiere, Final Cut, CapCut, JianYing, Audacity.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pip-install-blue.svg)](https://pypi.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![FunASR](https://img.shields.io/badge/powered%20by-FunASR-orange.svg)](https://github.com/modelscope/FunASR)

## What is subforge?

`subforge` is a subtitle-first Chinese ASR toolkit for **video editors, podcast producers, and content creators**. It wraps ModelScope's FunASR (Paraformer-large + VAD + punctuation + speaker diarization) and adds:

- **Multi-format subtitle export** — SRT / WebVTT / LRC, ready for import into any NLE
- **Multi-speaker diarization** — auto-detects who said what (great for interviews / podcasts / vlogs)
- **Built-in web UI** — drag-and-drop upload, see results, download all formats in one click
- **REST API + WebSocket** — drop into your existing pipeline or video editor plugin
- **Multiple models** — Paraformer-large (Chinese, highest quality), SenseVoice (multilingual, lightweight), streaming variant for live use
- **Pure-Python exporters** — zero external dependencies for the subtitle rendering itself

## Why?

Because clipping/transcribing by hand takes hours. subforge turns a 5-minute Chinese podcast into a 190-segment, 2-speaker, time-coded, punctuated transcript in ~15 seconds on an RTX 4070 Ti.

## Quick start

```bash
# 1. Install (requires conda env with torch+cu121 + funasr already set up)
pip install -e .

# 2. List registered models
asr models

# 3. Transcribe + export everything
asr transcribe interview.wav \
    --model paraformer-zh \
    --format srt,vtt,lrc,txt,md,json \
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

### 2. Multi-speaker diarization — automatic

subforge runs **speaker clustering** automatically. You don't pick K — the engine picks it for you based on eigengap heuristics over CAM++ speaker embeddings.

```bash
# Let the engine decide how many speakers
asr transcribe interview.wav --model paraformer-zh --format srt,json

# Force a specific count (e.g., "I know it's 3 people")
asr transcribe interview.wav --model paraformer-zh --spk-num 3
```

Under the hood: VAD → Paraformer (text + per-token timestamps) → CAM++ (192-dim speaker embedding per segment) → spectral clustering with eigengap for K → label back-fill.

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

# Transcribe (multipart upload)
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@interview.wav" \
  -F "model=paraformer-zh" \
  -F "formats=srt,vtt"
```

Returns JSON with `text`, `num_speakers`, `segments[]`, and (if requested) inline SRT/VTT/etc.

### 5. WebSocket streaming — live / real-time

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
asr models                              # list registered models
asr transcribe <audio>...               # run inference + export
    --model NAME                        # default: models.yaml's `default`
    --format srt,vtt,txt,md,lrc,json    # comma-separated; omit for stdout
    -o DIR                              # output dir (required if --format)
    --spk-num N                         # force speaker count
    --language auto|zh|en|yue|ja|ko     # multilingual models
    --batch-size-s N                    # override model's default
asr export <result.json>                # re-export from saved JSON (no re-inference)
    --format srt,vtt,...
    -o FILE                             # single file
    --output-dir DIR                    # multiple formats
asr server [--host 0.0.0.0] [--port 8000]
```

## Python API

```python
from subforge import load_registry, ModelRegistry, Transcriber

specs, default = load_registry()
registry = ModelRegistry(specs)
tr = Transcriber(registry.get("paraformer-zh"), specs["paraformer-zh"])

results = tr.transcribe("interview.wav")
r = results[0]
print(r.text)                # full transcript
print(r.num_speakers)        # auto-detected count
print(len(r.segments))       # 190 for a 5-min interview

# Save subtitles
from subforge import exporters
exporters.export(r, "srt", "out.srt")
exporters.export(r, "vtt", "out.vtt", speaker_labels=True)
```

## Registered models

| Name | ModelScope ID | Languages | Speakers | Streaming |
|---|---|---|---|---|
| `paraformer-zh` (default) | `iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn` | Chinese | ✅ auto | ❌ |
| `sensevoice` | `iic/SenseVoiceSmall` | zh/en/yue/ja/ko | ❌ | ❌ |
| `paraformer-zh-streaming` | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | Chinese | ❌ | ✅ |

Add your own in [`models.yaml`](models.yaml).

## How multi-speaker recognition works

This is **not** a single end-to-end model. It's a 5-step pipeline coordinated by FunASR:

```
audio.wav
  │
  ▼
[1] VAD (FSMN) → segments of "speech present"
  │
  ▼
[2] Paraformer-large → per-segment text + per-token timestamps
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

## Project structure

```
subforge/
├── pyproject.toml          # PEP 621, installable via pip
├── models.yaml             # model registry (edit me to add models)
├── subforge/               # Python package source
│   ├── __init__.py
│   ├── config.py           # load_registry() / ModelSpec
│   ├── models.py           # ModelRegistry (only AutoModel instantiation site)
│   ├── transcriber.py      # Transcribe + TranscriptResult normalization
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
- **Auto speaker count on short audio** — < 20 segments → defaults to single speaker. Force with `--spk-num`.
- **No CORS** — the server doesn't set CORS headers. Cross-origin browser clients need a CORS proxy, or deploy the UI on the same origin.
- **Windows console encoding** — Windows console is GBK by default; CLI auto-reconfigures to UTF-8. If you see garbled Chinese, set `PYTHONIOENCODING=utf-8`.

## Requirements

- Python 3.10+
- PyTorch with CUDA support (tested on 2.5.1+cu121)
- `funasr >= 1.3` and `modelscope >= 1.38` (these bring most other deps)
- `fastapi`, `uvicorn[standard]`, `python-multipart`, `websockets`, `pyyaml` (auto-installed by `pip install -e .`)
- ffmpeg (only for non-wav HTTP uploads)
- GPU: ≥ 8 GB VRAM recommended (RTX 4070 Ti 12 GB verified)

## License

MIT

## Acknowledgments

- [FunASR](https://github.com/modelscope/FunASR) — the underlying ASR / VAD / punctuation / speaker toolkit
- [ModelScope](https://www.modelscope.cn/) — model hosting
- [Paraformer](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn) — the workhorse ASR model
- [CAM++](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) — speaker embedding model