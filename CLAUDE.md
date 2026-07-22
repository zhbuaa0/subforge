# subforge — 项目冷启动

## 环境

```bash
conda activate paraformer-asr    # 必须！base 环境缺少依赖
# Python 3.10.20, torch 2.13.0, vllm 0.25.1, funasr 1.3.22
# 代码在 /home/zhbuaa0/subforge，已 pip install -e .
```

## 核心命令

| 命令 | 说明 |
|---|---|
| `asr transcribe input.wav -m paraformer-zh -o output/` | FunASR 本地推理 |
| `asr server --host 0.0.0.0 --port 8002 --log-level info` | 启动 subforge 服务 |
| `asr models` | 列出 6 个已注册模型 |
| `vllm serve ...` | 启动 MOSS vLLM 服务（另一终端）|

## 已知问题：vLLM 后端 404

- `vllm_backend.py` 发 POST 到 `/v1/chat/completions`
- 但当前 vLLM 只为 MOSS 暴露 `/v1/audio/transcriptions`（无 chat completions 路由）
- 可用 `/v1/audio/transcriptions` 替代，或使用 `moss-transcribe-diarize`（HF 后端，慢但可用）

## 端口

- subforge: 8002
- vLLM: 8001
