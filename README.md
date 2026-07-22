# subforge

> **Subtitle forge from Chinese ASR — dual backends: FunASR + MOSS.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pip-install-blue.svg)](https://pypi.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![FunASR](https://img.shields.io/badge/powered%20by-FunASR-orange.svg)](https://github.com/modelscope/FunASR)
[![MOSS](https://img.shields.io/badge/powered%20by-MOSS--Transcribe--Diarize-purple.svg)](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)

**📖 中文文档是完整版本 → [README.zh.md](README.zh.md)**

This project is primarily documented in Chinese. English users please see below for a quick overview.

---

# 🧊 Cold Start Recovery

> For Claude to resume project state without conversation context.

## Environment

```bash
conda activate paraformer-asr    # MUST: base env lacks all dependencies
# Python 3.10.20, torch 2.13.0, vllm 0.25.1, funasr 1.3.22
# subforge already `pip install -e .` in paraformer-asr env
```

## Service Ports

| Service | Port | Usage |
|---|---|---|
| subforge FastAPI + Web UI | **8002** (8000 was busy) | ASR transcription REST API |
| vLLM (MOSS) | **8001** | OpenAI-compatible inference |
| Legacy paraformer-asr | **8099** | standalone (ignore) |

## Startup

```bash
# 1. Start subforge server (binds 0.0.0.0, CORS enabled)
cd /home/zhbuaa0/subforge
asr server --host 0.0.0.0 --port 8002 --log-level info

# 2. Start vLLM (separate terminal, for MOSS inference)
# NOTE: --max-model-len 16384 limits input+output tokens; long audio truncates.
# Increase to 32768 or 65536 if you have more VRAM.
vllm serve /home/zhbuaa0/MOSS-Transcribe-Diarize/model_weights \
    --trust-remote-code --port 8001 --host 0.0.0.0 \
    --gpu-memory-utilization 0.75 --max-model-len 16384 \
    --max-num-batched-tokens 12288 --enforce-eager \
    --served-model-name OpenMOSS-Team/MOSS-Transcribe-Diarize

# 3. Local FunASR inference (no vLLM needed)
asr transcribe input.wav -m paraformer-zh -o output/
```

## Quick Test

```bash
# Check vLLM health
curl http://127.0.0.1:8001/health

# Check subforge server
curl http://127.0.0.1:8002/health
curl http://127.0.0.1:8002/models

# Submit transcription job
curl -X POST http://127.0.0.1:8002/transcribe \
  -F "file=@/path/to/audio.wav" \
  -F "model=paraformer-zh" \
  -F "formats=srt,json"
```

## Registered Models (6)

| Name | Backend | Streaming | Speakers | Multilingual | Use case |
|---|---|---|---|---|---|
| `paraformer-zh` (default) | FunASR | no | auto | no | General ASR |
| `seaco-paraformer-zh` | FunASR | no | overlapped | no | Meetings |
| `sensevoice` | FunASR | no | no | zh/en/yue/ja/ko | Multilingual |
| `paraformer-zh-streaming` | FunASR | yes | no | no | Live captions |
| `moss-transcribe-diarize` | MOSS (HF) | no | auto (e2e) | 50+ langs | Long meetings |
| `moss-transcribe-diarize-vllm` | vLLM | no | auto (e2e) | 50+ langs | Same MOSS, fast |

## Known Issue: vLLM Truncation

vLLM's `--max-model-len` limits **input + output tokens combined**. Long audio consumes many input tokens, leaving less for output → truncation. Fix: restart vLLM with larger `--max-model-len` (e.g. 32768 or 65536, if VRAM allows).

---



---

## TL;DR

`subforge` is a subtitle-first Chinese ASR toolkit for **video editors, podcast producers, and content creators**. It supports two backends:

- **FunASR backend** — wraps ModelScope's FunASR (Paraformer-large + VAD + punctuation + CAM++ speaker diarization)
- **MOSS backend** — wraps [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), a 0.9B end-to-end transformer for joint ASR + diarization

### Quick start

```bash
pip install -e .
asr transcribe interview.wav --format srt,vtt,txt -o output/
asr server --host 0.0.0.0 --port 8000   # web UI
```

### Key features

| Feature | Description |
|---|---|
| Subtitle export | SRT / VTT / LRC / TXT / MD / JSON |
| Diarization | Pipeline (FunASR) or end-to-end (MOSS) |
| Web UI | Drag-and-drop upload, all formats, one click |
| API | REST + WebSocket streaming |
| AI Polish | LLM-based light cleanup (Minimax, DeepSeek, OpenAI) |

### Requirements

- Python 3.10+, PyTorch with CUDA, GPU ≥ 8 GB VRAM
- **FunASR backend:** `funasr >= 1.3`, `modelscope >= 1.38`
- **MOSS backend:** `transformers >= 5.0`, `moss-transcribe-diarize >= 0.1.0`
- `fastapi`, `uvicorn`, `websockets`, `pyyaml`

### Registered models

| Name | Backend | Languages | Speakers | Streaming |
|---|---|---|---|---|
| `paraformer-zh` (default) | FunASR | Chinese | auto (pipeline) | no |
| `seaco-paraformer-zh` | FunASR | Chinese | overlapped speech | no |
| `sensevoice` | FunASR | zh/en/yue/ja/ko | no | no |
| `paraformer-zh-streaming` | FunASR | Chinese | no | **yes** |
| `moss-transcribe-diarize` | **MOSS** | 50+ languages | auto (e2e) | no |

### Deployment

| Scenario | Guide |
|---|---|
| **Windows + WSL2 (recommended for RTX GPUs)** | See [docs/DEPLOYMENT.md §1](docs/DEPLOYMENT.md#1-wsl2-部署推荐-windows-用户) |
| **Linux server (production, includes vLLM)** | See [docs/DEPLOYMENT.md §2](docs/DEPLOYMENT.md#2-linux-服务器原生部署) |
| **MOSS speedup (vLLM / sdpa / Flash-Attn)** | See [docs/DEPLOYMENT.md §3](docs/DEPLOYMENT.md#3-moss-加速方案vllm--sdpa) |
| **Quick `pip install`** | See [requirements/README.md](requirements/README.md) |

Python requirements are organized by backend:

```bash
# CPU-only (no GPU)
pip install -r requirements/base.txt -r requirements/funasr.txt -r requirements/moss.txt

# GPU: install torch first (pick your CUDA from pytorch.org),
# then the Python deps:
pip install -r requirements/base.txt -r requirements/funasr.txt -r requirements/moss.txt
```

Or with `pyproject.toml` extras:

```bash
pip install -e ".[moss-runtime]"        # base + MOSS backend
pip install -e ".[gpu-cu121]"          # everything for CUDA 12.1
pip install -e ".[cpu-all]"             # everything, no GPU
```

### License

MIT

### Acknowledgments

[FunASR](https://github.com/modelscope/FunASR) · [ModelScope](https://www.modelscope.cn/) · [Paraformer](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn) · [CAM++](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) · [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)