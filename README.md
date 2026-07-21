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