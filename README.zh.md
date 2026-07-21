# subforge

> **字幕工坊 —— 双引擎 ASR 中文语音识别 + 字幕工具。**
> 把音频一键转成可编辑字幕 —— SRT / VTT / LRC 直接导入 Premiere、Final Cut、剪映、CapCut、Audacity。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pip-install-blue.svg)](https://pypi.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![FunASR](https://img.shields.io/badge/powered%20by-FunASR-orange.svg)](https://github.com/modelscope/FunASR)
[![MOSS](https://img.shields.io/badge/powered%20by-MOSS--Transcribe--Diarize-purple.svg)](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)

## subforge 是什么？

`subforge` 是一款 **面向视频剪辑师 / 播客制作人 / 内容创作者** 的字幕优先中文 ASR 工具包，支持两个独立后端：

- **FunASR 后端**（默认）—— 封装 ModelScope 的 FunASR（Paraformer-large + VAD + 标点 + CAM++ 说话人分离）
- **MOSS 后端** —— 封装 [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)，一个 0.9B 端到端 transformer 模型，单次前向传播同时完成转写和说话人分离

两个后端共用同一套 CLI 标志、REST API、字幕导出器和 Web UI，通过 `--model` 切换。

特性：

- **多格式字幕导出** —— SRT / WebVTT / LRC，导入任何剪辑软件即用
- **多人对话说话人分离** —— 自动识别谁说的（适合访谈 / 播客 / vlog）
- **内置 Web UI** —— 拖拽上传、看结果、一键下载所有格式
- **REST API + WebSocket** —— 接入你现有的 pipeline 或剪辑插件
- **多后端支持** —— Paraformer-large（VAD/标点/spk 全套）、SenseVoice（多语种、轻量）、流式版、*以及* MOSS-Transcribe-Diarize（端到端、长会议、播客）
- **纯 Python 字幕导出** —— 字幕渲染本身零外部依赖

## 为什么做这个？

因为手工剪字幕要花几小时。subforge 把一段 5 分钟的中文播客，在 RTX 4070 Ti 上 ~15 秒变成 190 段、2 个说话人、带时间码、带标点的转录文本。

## 快速开始

```bash
# 1. 安装（需要已配好 conda env：torch+cu121 + funasr 或 transformers）
pip install -e .

# 2. 列已注册模型
asr models

# 3a. FunASR 推理（默认）
asr transcribe interview.wav \
    --model paraformer-zh \
    --format srt,vtt,lrc,txt,md,json \
    -o output/

# 3b. MOSS 推理（端到端 ASR + 说话人分离，适合长会议）
asr transcribe meeting.wav \
    --model moss-transcribe-diarize \
    --max-new-tokens 65536 \
    --format srt,json,txt \
    -o output/

# 4. （可选）启动 Web UI
asr server --host 0.0.0.0 --port 8000
#  → 浏览器打开 http://localhost:8000/
```

## 详细功能

### 1. 字幕导出 —— 6 种格式，零外部依赖

| 格式 | 用途 | 导入到 |
|---|---|---|
| **SRT** | 通用字幕标准 | Premiere、Final Cut、DaVinci、剪映、CapCut、Aegisub |
| **WebVTT** | 网页视频（`<track>` 元素）| HTML5 `<video>`、YouTube 字幕、OBS |
| **LRC** | 带时间戳的歌词格式 | 音乐播放器、卡拉 OK |
| **TXT** | 带时间戳的纯阅读文本 | 任何编辑器 |
| **MD** | Markdown 表格，便于 review | GitHub PR、Notion |
| **JSON** | FunASR 原始 schema，便于二次导出 | 程序化处理 |

说话人标签是一等公民：SRT 用 `Speaker 1:` 前缀，VTT 用原生 `<v Speaker 1>` 语音标签。

### 2. 多人说话人分离 —— 两种方案

subforge 支持两种完全不同的说话人分离策略，由你选的模型自动决定：

| 方案 | 模型 | 原理 |
|---|---|---|
| **Pipeline**（FunASR） | `paraformer-zh`, `seaco-paraformer-zh` | VAD → ASR → CAM++ 嵌入 → 谱聚类 + eigengap 选 K |
| **端到端**（MOSS） | `moss-transcribe-diarize` | 单 transformer 直接生成 `[start][Sxx]...text...[end]` —— 转写 + 分离一步完成 |

```bash
# Pipeline：让引擎自动决定说话人数
asr transcribe interview.wav --model paraformer-zh --format srt,json

# Pipeline：强制指定人数（"我知道是 3 个人"）
asr transcribe interview.wav --model paraformer-zh --spk-num 3

# 端到端：MOSS 自动检测说话人数（不支持 --spk-num）
asr transcribe meeting.wav --model moss-transcribe-diarize --format srt,json
```

底层流程（FunASR pipeline）：VAD → Paraformer（文本 + 每 token 时间戳）→ CAM++（每段 192 维说话人嵌入）→ 谱聚类 + eigengap 选 K → 回填标签。

### 3. Web UI —— 无需命令行

`asr server` 自带单页 HTML UI：

- 拖拽上传
- 模型下拉（自动从 `/models` 拉）
- 格式多选框
- 说话人数覆盖
- 实时统计：模型、说话人数、段数、文本长度
- 每格式独立下载按钮

启动 server 后浏览器打开 `http://<your-host>:8000/` 即可。

### 4. REST API —— 程序化访问

```bash
# 健康检查
curl http://localhost:8000/health

# 列模型
curl http://localhost:8000/models

# 转写（multipart 上传，FunASR）
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@interview.wav" \
  -F "model=paraformer-zh" \
  -F "formats=srt,vtt"

# 转写（multipart 上传，MOSS）
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@meeting.wav" \
  -F "model=moss-transcribe-diarize" \
  -F "formats=srt,json"
```

返回 JSON 含 `text` / `num_speakers` / `segments[]` / （如果请求）内嵌的 SRT / VTT 等。

### 5. Python 代码调用服务端

```python
import httpx
from pathlib import Path

# ── 列模型 ──
r = httpx.get("http://localhost:8000/models")
print(r.json()["default"])               # "paraformer-zh"

# ── FunASR 转写（快速）──
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

# ── MOSS 转写（端到端，较慢）──
with open("meeting.wav", "rb") as f:
    r = httpx.post(
        "http://localhost:8000/transcribe",
        data={"model": "moss-transcribe-diarize", "formats": "srt,json"},
        files={"audio": f},
    )
data = r.json()
print(f"{data['num_speakers']} 个说话人, {len(data['segments'])} 段")
```

### 6. WebSocket 流式 —— 实时

实时转写（访谈录制、播客录音、直播字幕）：

```python
import websockets, json, asyncio

async def main():
    async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
        await ws.send(json.dumps({"model": "paraformer-zh-streaming"}))
        ready = json.loads(await ws.recv())
        print(ready)  # {"status": "ready", ...}

        # 发 wav 字节；服务端回 partial / final
        with open("interview.wav", "rb") as f:
            await ws.send(f.read())
        msg = json.loads(await ws.recv())
        print(msg["partial"], msg["text"])

        await ws.send(json.dumps({"action": "final"}))
        print(json.loads(await ws.recv())["text"])

asyncio.run(main())
```

## CLI 参考

```
asr models                              # 列已注册模型（显示 BACKEND 列）
asr transcribe <audio>...               # 推理 + 导出
    --model NAME                        # 默认：models.yaml 的 default
    --format srt,vtt,txt,md,lrc,json    # 逗号分隔；省略则只打印到 stdout
    -o DIR                              # 输出目录（与 --format 配合时必需）
    --spk-num N                         # 强制说话人数（仅 FunASR；MOSS 忽略此参数）
    --language auto|zh|en|yue|ja|ko     # 多语种模型
    --batch-size-s N                    # 覆盖模型默认值（FunASR）
    --max-new-tokens N                  # MOSS 生成长度上限（默认 65536）
asr export <result.json>                # 从已有 JSON 重新导出（不重跑推理）
    --format srt,vtt,...
    -o FILE                             # 单文件输出
    --output-dir DIR                    # 多格式目录输出
asr polish <result.json>                # AI 润色（任意的后后端都支持）
    --provider minimax|deepseek|openai
    --format srt,vtt,txt,md
asr server [--host 0.0.0.0] [--port 8000]
```

## Python API

```python
from subforge import load_registry, ModelRegistry, Transcriber

specs, default = load_registry()
registry = ModelRegistry(specs)

# FunASR（pipeline 后端）
tr = Transcriber(registry.get("paraformer-zh"), specs["paraformer-zh"])
results = tr.transcribe("interview.wav")
r = results[0]
print(r.text)                # 完整文本
print(r.num_speakers)        # 自动检测的说话人数
print(len(r.segments))       # 5 分钟访谈约 190 段

# MOSS（端到端后端）
tr2 = Transcriber(registry.get("moss-transcribe-diarize"), specs["moss-transcribe-diarize"])
r2 = tr2.transcribe("meeting.wav", max_new_tokens=65536)[0]
print(f"{r2.num_speakers} 个说话人, {len(r2.segments)} 段")

# 写字幕（同一组 API，不依赖后端）
from subforge import exporters
exporters.export(r2, "srt", "out.srt")
exporters.export(r2, "vtt", "out.vtt", speaker_labels=True)
```

## 已注册模型

| 名字 | 后端 | 模型 ID | 语种 | 说话人 | 流式 | 适用场景 |
|---|---|---|---|---|---|---|
| `paraformer-zh`（默认）| FunASR | `iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn` | 中文 | 自动 | 否 | 通用 ASR |
| `seaco-paraformer-zh` | FunASR | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 中文 | 否* | 否 | 重叠语音 |
| `sensevoice` | FunASR | `iic/SenseVoiceSmall` | zh/en/yue/ja/ko | 否 | 否 | 多语种 |
| `paraformer-zh-streaming` | FunASR | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | 中文 | 否 | 是 | 直播字幕 |
| `moss-transcribe-diarize` | **MOSS** | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | 50+ 语言 | 自动（端到端）| 否 | 长会议、播客 |

* SeACo 内部处理多说话人，spk 标签不一定可靠。

在 [`models.yaml`](models.yaml) 加你自己的模型。设 `backend: moss` 使用 MOSS 适配器，省略则默认 `funasr`。

## 多人对话识别是怎么做的？

FunASR pipeline 由 5 步组成。 MOSS-Transcribe-Diarize 不同 —— 它在单次 transformer 前向传播中直接生成 `[start][Sxx]...text...[end]`（见下文 [MOSS 端到端分离](#moss-端到端说话人分离)）。

```
audio.wav
  │
  ▼
[1] VAD (FSMN) → 切出"有语音"的段
  │
  ▼
[2] FunASR ASR → 每段文本 + 每 token 时间戳
  │
  ▼
[3] CAM++ → 每段 192 维说话人嵌入
  │
  ▼
[4] 谱聚类 + eigengap → 选 K、回填说话人标签
  │
  ▼
[5] 输出 sentence_info[{start, end, text, spk, timestamp}]
```

传 `--spk-num N` 时，第 4 步强制 K=N 而不是自动检测。短音频（< 20 段）会自动归为单说话人；想强制多人请加 `--spk-num`。

### MOSS 端到端说话人分离

[MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) 是一个 0.9B transformer，**单次前向传播**同时完成转写和说话人分离。与上述 5 步 pipeline 不同，它：

- 通过 Whisper 风格的编码器一次性编码整段音频
- 自回归生成 `[start<时间戳>][S<说话人>]转写文本[end<时间戳>]` token
- 支持 50+ 语言（Whisper 词表），不需要单独的 VAD 或说话人嵌入模型

**何时用 MOSS：**

| 场景 | 推荐 | 理由 |
|---|---|---|
| 长会议（15–60 分钟） | MOSS | 无 VAD chunk 限制；一次完成说话人分离 |
| 2–4 人播客 | MOSS | 说话人分离比 pipeline 更清晰 |
| 重叠语音 / 抢话 | MOSS | 端到端模型处理重叠更好 |
| 快速转写（<10 分钟短音频） | FunASR | ~15 秒 vs MOSS ~5 分钟（RTX 4070 Ti） |
| 流式 / 实时 | FunASR | MOSS 不支持流式 |

**限制：**
- 5 分钟音频生成约 3000 token（12 GB 显存）；超长音频（>60 分钟）可能碰 `max_new_tokens` 上限
- 不支持 `--spk-num` —— 说话人数由模型自动检测
- 5 分钟音频推理约 5–6 分钟（RTX 4070 Ti），FunASR pipeline 约 15 秒
- 需要 HuggingFace `transformers`；首次使用下载权重约 1.8 GB

## 项目结构

```
subforge/
├── pyproject.toml          # PEP 621，可 pip install -e .
├── models.yaml             # 模型注册表（编辑这里加模型）
├── subforge/               # Python 包源码
│   ├── __init__.py
│   ├── backends/            # ASR 模型后端
│   │   ├── __init__.py
│   │   ├── base.py          # BaseBackend 协议
│   │   ├── funasr_backend.py # FunASR AutoModel 适配器（默认）
│   │   └── moss_backend.py  # MOSS-Transcribe-Diarize 适配器
│   ├── config.py           # load_registry() / ModelSpec
│   ├── models.py           # ModelRegistry（按 backend 分发）
│   ├── transcriber.py      # Transcriber（委托给 backend）+ TranscriptResult 归一化
│   ├── exporters.py        # 纯 stdlib 字幕导出器
│   ├── audio.py            # 上传 + ffmpeg 回退
│   ├── streaming.py        # WebSocket StreamingSession
│   ├── server.py           # FastAPI app + lifespan + 路由
│   ├── cli.py              # argparse 子命令
│   └── web/
│       └── index.html      # 单页 Web UI
├── legacy/                 # 旧脚本的兼容 shim
├── scripts/                # 验证脚本（verify_phase1..6.py）
├── output/                 # 示例输出
├── README.md               # 英文（项目主页）
├── README.zh.md            # 中文（本文）
├── EXTENSION_PLAN.md       # 开发路线图 / 变更日志
└── LICENSE
```

## 角落案例与已知限制

- **ffmpeg** —— HTTP 上传非 wav/flac 必需。本地 CLI 处理 wav/flac 无依赖。安装：`conda install -c conda-forge ffmpeg`。
- **流式模型 RTF** —— `paraformer-zh-streaming` 在长音频上比批量慢 ~30×（短片段 RTF ~1.0，5 分钟整段 RTF ~120）。基础设施已就位；速度取舍是 FunASR 模型本身的特性。
- **MOSS 后端推理速度** —— 5 分钟音频推理约 5–6 分钟（RTX 4070 Ti），模型 ~RTF 1.0。短音频建议用 `paraformer-zh`（同 5 分钟 ~15 秒）。MOSS 的优点是长会议场景下单次推理就完成说话人分离。
- **MOSS 首次下载** —— 首次使用从 HuggingFace Hub 下载约 1.8 GB safetensors，缓存在 `~/.cache/huggingface/hub/`。
- **MOSS 不支持 `--spk-num`** —— 说话人数由模型自动检测。用 `--max-new-tokens` 控制生成长度上限（默认 65536）。
- **未开 CORS** —— 服务不设 CORS 头。跨域浏览器客户端需要 CORS 代理，或把 UI 部署在同一源。
- **Windows 控制台编码** —— Windows 控制台默认 GBK；CLI 自动重配为 UTF-8。如果还看到中文乱码，设 `PYTHONIOENCODING=utf-8`。

## 依赖

- Python 3.10+
- PyTorch CUDA（测过 2.5.1+cu121）
- **FunASR 后端**需要 `funasr >= 1.3` 和 `modelscope >= 1.38`（建议 conda 安装）
- **MOSS 后端**需要 `transformers >= 5.0` 和 `moss-transcribe-diarize >= 0.1.0`（`pip install subforge[moss-runtime]`；torch 走 conda 管理）
- `fastapi` / `uvicorn[standard]` / `python-multipart` / `websockets` / `pyyaml`（`pip install -e .` 自动装）
- ffmpeg（仅 HTTP 非 wav 上传）
- GPU：建议 ≥ 8 GB 显存（实测 RTX 4070 Ti 12 GB）

## 许可

MIT

## 部署

| 场景 | 指南 |
|---|---|
| **Windows + WSL2 (推荐家用 RTX)** | 跳 [docs/DEPLOYMENT.md §1](docs/DEPLOYMENT.md#1-wsl2-部署推荐-windows-用户) |
| **Linux 服务器 (生产，含 vLLM)** | 跳 [docs/DEPLOYMENT.md §2](docs/DEPLOYMENT.md#2-linux-服务器原生部署) |
| **MOSS 加速 (vLLM / sdpa / Flash-Attn)** | 跳 [docs/DEPLOYMENT.md §3](docs/DEPLOYMENT.md#3-moss-加速方案vllm--sdpa) |
| **快速 pip install** | 看 [requirements/README.md](requirements/README.md) |

Python 依赖按后端拆分:

```bash
# 仅 CPU 模式(不需要 GPU)
pip install -r requirements/base.txt -r requirements/funasr.txt -r requirements/moss.txt

# GPU: 先装 torch(从 pytorch.org 选你的 CUDA 版本)
#     然后再装 Python 依赖:
pip install -r requirements/base.txt -r requirements/funasr.txt -r requirements/moss.txt
```

或者用 `pyproject.toml` extras:

```bash
pip install -e ".[moss-runtime]"        # base + MOSS 后端
pip install -e ".[gpu-cu121]"          # CUDA 12.1 全部
pip install -e ".[cpu-all]"             # 全部, 不用 GPU
```

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) —— 底层 ASR / VAD / 标点 / 说话人工具
- [ModelScope](https://www.modelscope.cn/) —— FunASR 模型托管
- [Paraformer](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn) —— 主力 ASR 模型
- [CAM++](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) —— 说话人嵌入模型
- [MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) —— 端到端转写 + 说话人分离模型（0.9B transformer）