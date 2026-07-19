# Plan: 把 `paraformer-asr` 扩展为多模型 ASR 工具包

> 本地记录。完整版亦在 `C:\Users\zhbuaa0\.claude\plans\splendid-launching-stardust.md`。

## 实施进度（2026-07-19）

| Phase | 描述 | 状态 |
|---|---|---|
| 1.1 | 包骨架（pyproject + models.yaml + 5 个模块） | ✅ |
| 1.2 | 验证：复现 C3142.wav 旧结果（190 段、num_speakers=2、text_len=1631） | ✅ PASS |
| 2.1 | exporters.py：srt/vtt/lrc/txt/md/json 6 格式 | ✅ |
| 2.2 | cli.py：models / transcribe / export / server 子命令 | ✅ |
| 2.3 | legacy/ 4 个 shim 兼容旧入口 | ✅ |
| 2.4 | 验证：6 格式输出 + asr export 复读 + shim 编译 | ✅ PASS |
| 3.1 | models.yaml 注册 sensevoice（多语种、轻量）+ feature-gating | ✅ |
| 3.2 | 验证：rich_transcription 剥离 `<|zh|>` `<|HAPPY|>` `<|Speech|>` 等标签 | ✅ PASS |
| 4 | num_speakers 字段 + --spk-num flag（角落案例在 README 注明） | ✅ |
| 5.1 | audio.py：上传字节 → temp 文件 + ffmpeg 检测 | ✅ |
| 5.2 | server.py：FastAPI app + lifespan + /health /models /transcribe | ✅ |
| 5.3 | 验证：HTTP 接口 + feature-gating + 404 | ✅ PASS |
| 6.1 | models.yaml 注册 paraformer-zh-streaming + streaming.py | ✅ |
| 6.2 | server.py：/ws/stream 端点 + per-conn cache | ✅ |
| 6.3 | 验证：WS pipeline + 多连接 cache 隔离 | ✅ PASS（流式模型 RTF 高是模型问题，非包装问题） |

**全部 6 阶段完成。** 见 [README.md](README.md) 获取最终使用说明。

## Context

当前 `paraformer-asr` 是一个 **扁平脚本工作区**（4 个 `.py` 文件 + 一个 README，全部平铺在仓库根目录，无包、无测试、无 CI、无 license），封装 FunASR 的 Paraformer-large 做中文 ASR。脚本之间大量重复（同样的 4 个 ModelScope 模型 ID 出现在 `demo.py` / `transcribe.py` / `run_clean.py`），所有"参数"都是模块顶部硬编码常量。

用户希望在保留现有能力（中文离线 ASR + VAD + 标点 + 说话人分离）的基础上扩展 4 个方向：

1. **多模型支持** —— 通过配置切换 ASR 模型（如 Paraformer-large / SenseVoiceSmall），不需要改代码
2. **字幕导出** —— 除现有 txt/json/md 外，输出 SRT / WebVTT / LRC
3. **自动说话人数检测** —— 去掉硬编码的 `spk_num`，由引擎自动估计
4. **服务模式** —— 长期运行的 FastAPI + WebSocket 服务（单实例模型复用）

**关键发现**（来自 FunASR 源码调研）：
- **自动说话人检测已内置** —— `funasr/models/campplus/cluster_backend.py` 的 `ClusterBackend` 用谱聚类 + eigengap 自动选 k，仅在调用 `generate()` 时不传 `preset_spk_num` 即可。无需自实现聚类。
- **流式需要专门模型** —— 必须用 `*-online` 模型 + 每连接独立的 `cache` dict；批处理模型不能直接分块。
- **SenseVoice 不带 vad/punc/spk 子模型** —— 配置必须支持"无子模型"的模型，且通过 `rich_transcription_postprocess` 剥离 `<|zh|>` `<|EMO|>` 等标签。

---

## 目标目录结构

```
paraformer-asr/
├── pyproject.toml              # NEW — 包元信息 + 依赖 + `asr` CLI 入口
├── README.md                   # 更新使用说明
├── models.yaml                 # NEW — 模型注册表（友好名 → ModelSpec）
├── subforge/                   # NEW 包（重命名自 paraformer_asr）
│   ├── __init__.py             # 版本、对外 re-export
│   ├── config.py               # 解析 models.yaml → ModelSpec dataclass
│   ├── models.py               # ModelRegistry：唯一调用 AutoModel(...) 的地方
│   ├── transcriber.py          # Transcriber：统一 generate() + 后处理
│   ├── exporters.py            # TranscriptResult → srt / vtt / lrc / txt / md / json（纯 stdlib）
│   ├── audio.py                # 输入标准化（ffmpeg / librosa → 16k mono f32）
│   ├── streaming.py            # 流式会话（cache 管理）
│   ├── server.py               # FastAPI：/transcribe / /models / /ws/stream
│   └── cli.py                  # argparse 子命令
├── legacy/                     # MOVED 旧脚本，改为薄 shim
│   ├── demo.py  transcribe.py  run_clean.py  make_md.py
└── output/                     # gitignore 的产物目录
```

---

## 关键文件与用途

| 文件 | 职责 | 是否新增 |
|---|---|---|
| `pyproject.toml` | PEP 621 元数据；`[project.scripts] asr = "subforge.cli:main"` | 新 |
| `models.yaml` | 模型注册表（见下） | 新 |
| `subforge/config.py` | 加载校验 YAML；导出 `ModelSpec` dataclass | 新 |
| `subforge/models.py` | `ModelRegistry.build(name, device)`；**唯一**实例化 `funasr.AutoModel` 的地方；按 name 缓存 | 新 |
| `subforge/transcriber.py` | `Transcriber.transcribe(audio, preset_spk_num=None, **overrides) -> TranscriptResult`；将 FunASR 的 `sentence_info` 归一化为 `Segment[]`；SenseVoice 调用 `funasr.utils.postprocess_utils.rich_transcription_postprocess` | 新 |
| `subforge/exporters.py` | `to_srt/to_vtt/to_lrc/to_txt/to_md/to_json(r) -> str`；`export(r, fmt, path)`；纯 stdlib，约 30 行/格式；VTT 用原生 `<v Speaker 1>文本</v>`，SRT 用 `Speaker 1: 文本` 前缀 | 新 |
| `subforge/audio.py` | `load_audio(src: str|bytes|Path) -> np.ndarray`；文件/URL 直接交给 FunASR；原始字节（WS / upload）走 ffmpeg → soundfile；缺 ffmpeg 时给出明确报错 | 新 |
| `subforge/streaming.py` | `StreamingSession(model, spec)`；维护 `self.cache`；`push(pcm_chunk, is_final)` 调用 `model.generate(input=chunk, cache=self.cache, is_final=is_final, **spec.generate)` 返回增量文本 | 新 |
| `subforge/server.py` | FastAPI app + `lifespan` 一次性加载模型；`asyncio.Lock` 串行化 GPU；`POST /transcribe` 跑在 `run_in_threadpool`；`WS /ws/stream` 每连接一个 `StreamingSession` | 新 |
| `subforge/cli.py` | argparse 子命令：`asr transcribe | export | models | server` | 新 |
| `legacy/{demo,transcribe,run_clean,make_md}.py` | 薄 shim，内部调用新 CLI，保留旧命令可用 | 移动 |

### `models.yaml`（完整示例）

```yaml
default: paraformer-zh

models:
  paraformer-zh:                       # 当前默认：完整离线管线
    model: iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn
    vad_model:  iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
    punc_model: iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch
    spk_model:  iic/speech_campplus_sv_zh-cn_16k-common
    init:    { disable_update: true }
    generate: { batch_size_s: 300 }     # preset_spk_num 不传 ⇒ 自动检测
    features: { timestamps: true, speakers: true, punc: true }

  sensevoice:                          # 多语种、轻量、无 vad/punc/spk
    model: iic/SenseVoiceSmall
    init: { disable_update: true }
    generate: { language: auto, use_itn: true, batch_size_s: 300 }
    postprocess: rich_transcription    # 剥离 <|zh|><|EMO|> 等标签
    features: { timestamps: true, speakers: false, punc: true, multilingual: true }

  paraformer-zh-streaming:             # 仅 WS 使用
    model: iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
    init: { disable_update: true }
    streaming: true
    generate: { chunk_size: [0, 10, 5], encoder_chunk_look_back: 4, decoder_chunk_look_back: 1 }
    features: { timestamps: false, speakers: false, streaming: true }
```

`config.load_registry()` 校验：流式模型禁止走 `/transcribe`；非流式禁止走 `/ws/stream`；`spk_model` 缺失时，调用 `--spk-num` / 选 spk 输出给出明确错误（处理 SenseVoice "优雅禁用" 场景）。

### `TranscriptResult` 数据结构

```python
@dataclass
class Segment:
    start: float          # 秒
    end: float            # 秒
    text: str
    spk: int | None       # None ⇒ 未知（SenseVoice）

@dataclass
class TranscriptResult:
    text: str                                  # 全文本
    segments: list[Segment]
    num_speakers: int | None                   # = max(spk)+1，spk 缺失则 None
    language: str | None                       # SenseVoice 才填
    raw: dict | None                           # 原始 FunASR 输出
```

---

## CLI 用法示例

```bash
# 列出已注册模型
asr models

# 转写并多格式输出
asr transcribe meeting.wav --model paraformer-zh --format srt,vtt,txt -o output/

# 多语种（自动检测语言）
asr transcribe talk.mp3 --model sensevoice --format srt

# 强制说话人数（绕过自动检测）
asr transcribe call.wav --model paraformer-zh --spk-num 2 --format json

# 重新导出（不重跑推理）
asr export output/meeting.json --format lrc -o meeting.lrc

# 启服务（HTTP + WebSocket）
asr server --host 0.0.0.0 --port 8000
```

---

## 服务模式接口

| 路由 | 方法 | 说明 |
|---|---|---|
| `/models` | GET | 列出注册表（name + features） |
| `/transcribe` | POST | multipart 上传音频 + form（model, formats[], preset_spk_num, language）；返回 JSON + 内嵌文件 |
| `/ws/stream` | WS | 客户端逐块发 PCM（16k mono f32），服务端回 `{partial: "...", final: bool}` JSON 帧；每连接独立 `StreamingSession` |

- **lifespan** 一次性构建默认模型 + 流式模型，缓存在 `app.state`，跨请求复用（避免每次加载要十几秒）
- **GPU 并发**：单 `asyncio.Lock` 串行化 `generate()`；通过 `await run_in_threadpool(...)` 跑阻塞推理，保持事件循环 + WS ping 响应
- **ffmpeg 依赖**：WS 客户端推送 webm/opus 时由 `audio.py` 调 ffmpeg；启动时 `check ffmpeg` 缺失则告警

---

## 要添加的依赖

- **运行时**：`fastapi`、`uvicorn[standard]`、`python-multipart`、`websockets`、`pyyaml`
- **已传递依赖**（funasr 自带，不重列）：`scikit-learn`、`scipy`、`librosa`、`soundfile`、`torchaudio`
- **不锁** torch / funasr 版本（CUDA 轮子环境相关，README 已注明）
- **系统依赖**：`ffmpeg`（PATH 中）—— 仅服务模式需要
- **打包**：`pyproject.toml`，**不**用 `requirements.txt`

---

## 复用现有代码

| 现有 | 落到哪里 | 说明 |
|---|---|---|
| `demo.py:8-11` 的模型 ID 常量 | `models.yaml` 的 `paraformer-zh` 条目 | 一字不改地搬过去 |
| `run_clean.py:46-48` 的 JSON dump 逻辑 | `exporters.to_json()` | 字段保留 `{text, sentence_info, timestamp}` |
| `run_clean.py` 输出 txt 的格式化（segment + 全文本 + 头部元信息） | `exporters.to_txt()` | 头部加模型名 / num_speakers / 时长 |
| `make_md.py` 整文件（27 行） | `exporters.to_md()` | 原样移植 |
| `demo.py` 的 `print(f"[{start:>6.2f}s - {end:>6.2f}s] spk{spk}: {text}")` | `exporters.to_txt(plain=True)` 默认行为 | 直接搬 |

---

## 实施阶段（PR-sized）

1. **包骨架 + 核心重构** —— `pyproject.toml`、`config.py`、`models.yaml`、`models.py`、`transcriber.py`。验证：用 `paraformer-zh` 复现当前 `run_clean.py` 对 `C3142.wav` 的输出，与现有 `C3142_result.json` 字段一致。
2. **导出器 + CLI** —— `exporters.py`（6 个格式）、`cli.py`（`transcribe/export/models`）。把 4 个旧脚本改为 `legacy/` 下的 shim。
3. **多模型** —— 注册 `sensevoice` + `rich_transcription` 后处理；feature-gating（spk-less 模型拒绝 spk 选项）。用一个非中文样本验证。
4. **自动说话人** —— 暴露 `num_speakers` 字段 + `--spk-num` 参数；README 注明"少于 20 段时 FunASR 默认单说话人"的角落案例。
5. **HTTP 服务** —— `server.py`（`/models`、`/transcribe`）、`audio.py`、`lifespan` 加载、`gpu_lock`、ffmpeg 检测。
6. **WebSocket 流式** —— `streaming.py`、`/ws/stream`、注册 `paraformer-zh-streaming` 模型；附带一个最小 HTML/JS 测试页。

每个阶段结束都能运行（阶段 1 不破坏现有行为，阶段 2 起逐步增加能力）。

---

## Verification

每个阶段跑以下检查，确认通过再进下一阶段：

1. **阶段 1（行为等价）**
   ```bash
   conda run -n paraformer-asr python -m subforge.cli transcribe C3142.wav --model paraformer-zh --format json -o /tmp/v1/
   diff <(jq -S '.[0].sentence_info' /tmp/v1/C3142_result.json) \
        <(jq -S '.[0].sentence_info' C3142_result.json)
   ```
   应无差异或仅有毫秒级浮点抖动。
2. **阶段 2（格式导出）**
   ```bash
   asr transcribe C3142.wav --format srt,vtt,lrc,txt,md,json -o /tmp/v2/
   # 人工 spot-check SRT（时间码格式 HH:MM:SS,mmm）/ VTT（WEBVTT 头 + HH:MM:SS.mmm）
   ```
3. **阶段 3（多模型）**
   ```bash
   asr transcribe samples/en.wav --model sensevoice --format json -o /tmp/v3/
   jq '.[0].text, .[0].sentence_info[0].text' /tmp/v3/en_result.json   # 应有 <|en|> 被剥离
   ```
4. **阶段 4（自动 spk）**
   ```bash
   asr transcribe C3142.wav --format json -o /tmp/v4/
   jq '.[0].num_speakers, [.sentence_info[].spk] | unique' /tmp/v4/C3142_result.json
   ```
5. **阶段 5（HTTP）**
   ```bash
   asr server --port 8000 &
   curl -s -F audio=@C3142.wav -F model=paraformer-zh http://localhost:8000/transcribe | jq '.num_speakers, (.segments|length)'
   curl -s http://localhost:8000/models | jq
   ```
6. **阶段 6（WS）**
   - 浏览器打开测试页；按住说话；观察 partial 文本滚动，松开后收到 final=true 帧
   - 关闭并重连；验证新连接得到独立 cache（不与旧连接串台）

如果阶段 1 不能复现现有输出，**停在那里**，回到 `run_clean.py` 排查差异（一般是 `disable_update=True` 是否生效 / `batch_size_s` 默认值差异 / `preset_spk_num` 默认行为差异），不要往后推进。

---

## 开放问题（默认已采纳，实施前可再确认）

1. 多语种范围：默认仅 SenseVoice（zh/en/yue/ja/ko）。如需 Whisper / 中英 Paraformer，告知后追加 `models.yaml` 条目
2. 流式说话人：默认 WS 只返回文本（不带 spk）。如必须流式说话人，需研究 FunASR-Nano，单独评估
3. 并发：默认单 GPU 串行。多人并发场景再考虑队列 + 背压
4. 部署：默认本地 + localhost。需要对外暴露时再加 CORS / 鉴权 / 限流
5. 配置位置：默认仓库根 `models.yaml`。如需用户级配置（`~/.config/paraformer-asr/`）告知
6. CLI 名：默认 `asr`。如有冲突可用 `pyffasr` / `pfasr` 等替代