# subforge

> **字幕工坊 —— 中文 ASR 驱动的剪辑师利器。**
> 把音频一键转成可编辑字幕 —— SRT / VTT / LRC 直接导入 Premiere、Final Cut、剪映、CapCut、Audacity。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/pip-install-blue.svg)](https://pypi.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![FunASR](https://img.shields.io/badge/powered%20by-FunASR-orange.svg)](https://github.com/modelscope/FunASR)

## subforge 是什么？

`subforge` 是一款 **面向视频剪辑师 / 播客制作人 / 内容创作者** 的字幕优先中文 ASR 工具包。它封装了 ModelScope 的 FunASR（Paraformer-large + VAD + 标点 + 说话人分离），并提供：

- **多格式字幕导出** —— SRT / WebVTT / LRC，导入任何剪辑软件即用
- **多人对话说话人分离** —— 自动识别谁说的（适合访谈 / 播客 / vlog）
- **内置 Web UI** —— 拖拽上传、看结果、一键下载所有格式
- **REST API + WebSocket** —— 接入你现有的 pipeline 或剪辑插件
- **多模型可选** —— Paraformer-large（中文最高质量）、SenseVoice（多语种轻量）、流式版（实时）
- **纯 Python 字幕导出** —— 字幕渲染本身零外部依赖

## 为什么做这个？

因为手工剪字幕要花几小时。subforge 把一段 5 分钟的中文播客，在 RTX 4070 Ti 上 ~15 秒变成 190 段、2 个说话人、带时间码、带标点的转录文本。

## 快速开始

```bash
# 1. 安装（需要已配好 conda env：torch+cu121 + funasr）
pip install -e .

# 2. 列已注册模型
asr models

# 3. 转写并导出全部格式
asr transcribe interview.wav \
    --model paraformer-zh \
    --format srt,vtt,lrc,txt,md,json \
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

### 2. 多人说话人分离 —— 自动

subforge 自动跑 **说话人聚类**。你不用选 K —— 引擎根据 CAM++ 嵌入向量的 eigengap 启发式自动选。

```bash
# 让引擎自动决定说话人数
asr transcribe interview.wav --model paraformer-zh --format srt,json

# 强制指定人数（"我知道是 3 个人"）
asr transcribe interview.wav --model paraformer-zh --spk-num 3
```

底层流程：VAD → Paraformer（文本 + 每 token 时间戳）→ CAM++（每段 192 维说话人嵌入）→ 谱聚类 + eigengap 选 K → 回填标签。

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

# 转写（multipart 上传）
curl -X POST http://localhost:8000/transcribe \
  -F "audio=@interview.wav" \
  -F "model=paraformer-zh" \
  -F "formats=srt,vtt"
```

返回 JSON 含 `text` / `num_speakers` / `segments[]` / （如果请求）内嵌的 SRT / VTT 等。

### 5. WebSocket 流式 —— 实时

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
asr models                              # 列已注册模型
asr transcribe <audio>...               # 推理 + 导出
    --model NAME                        # 默认：models.yaml 的 default
    --format srt,vtt,txt,md,lrc,json    # 逗号分隔；省略则只打印到 stdout
    -o DIR                              # 输出目录（与 --format 配合时必需）
    --spk-num N                         # 强制说话人数
    --language auto|zh|en|yue|ja|ko     # 多语种模型
    --batch-size-s N                    # 覆盖模型默认值
asr export <result.json>                # 从已有 JSON 重新导出（不重跑推理）
    --format srt,vtt,...
    -o FILE                             # 单文件输出
    --output-dir DIR                    # 多格式目录输出
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
print(r.text)                # 完整文本
print(r.num_speakers)        # 自动检测的说话人数
print(len(r.segments))       # 5 分钟访谈约 190 段

# 写字幕
from subforge import exporters
exporters.export(r, "srt", "out.srt")
exporters.export(r, "vtt", "out.vtt", speaker_labels=True)
```

## 已注册模型

| 名字 | ModelScope ID | 语种 | 说话人 | 流式 |
|---|---|---|---|---|
| `paraformer-zh`（默认）| `iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn` | 中文 | ✅ 自动 | ❌ |
| `sensevoice` | `iic/SenseVoiceSmall` | zh/en/yue/ja/ko | ❌ | ❌ |
| `paraformer-zh-streaming` | `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online` | 中文 | ❌ | ✅ |

在 [`models.yaml`](models.yaml) 加你自己的。

## 多人对话识别是怎么做的？

**不是**一个端到端模型。是 5 步 pipeline，由 FunASR 协调：

```
audio.wav
  │
  ▼
[1] VAD (FSMN) → 切出"有语音"的段
  │
  ▼
[2] Paraformer-large → 每段文本 + 每 token 时间戳
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

## 项目结构

```
subforge/
├── pyproject.toml          # PEP 621，可 pip install -e .
├── models.yaml             # 模型注册表（编辑这里加模型）
├── subforge/               # Python 包源码
│   ├── __init__.py
│   ├── config.py           # load_registry() / ModelSpec
│   ├── models.py           # ModelRegistry（唯一 AutoModel 实例化点）
│   ├── transcriber.py      # Transcribe + TranscriptResult 归一化
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
- **短音频自动说话人数** —— < 20 段默认单说话人。用 `--spk-num` 强制覆盖。
- **未开 CORS** —— 服务不设 CORS 头。跨域浏览器客户端需要 CORS 代理，或把 UI 部署在同一源。
- **Windows 控制台编码** —— Windows 控制台默认 GBK；CLI 自动重配为 UTF-8。如果还看到中文乱码，设 `PYTHONIOENCODING=utf-8`。

## 依赖

- Python 3.10+
- PyTorch CUDA（测过 2.5.1+cu121）
- `funasr >= 1.3` 和 `modelscope >= 1.38`（自带大部分依赖）
- `fastapi` / `uvicorn[standard]` / `python-multipart` / `websockets` / `pyyaml`（`pip install -e .` 自动装）
- ffmpeg（仅 HTTP 非 wav 上传）
- GPU：建议 ≥ 8 GB 显存（实测 RTX 4070 Ti 12 GB）

## 许可

MIT

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) —— 底层 ASR / VAD / 标点 / 说话人工具
- [ModelScope](https://www.modelscope.cn/) —— 模型托管
- [Paraformer](https://www.modelscope.cn/models/iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn) —— 主力 ASR 模型
- [CAM++](https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common) —— 说话人嵌入模型