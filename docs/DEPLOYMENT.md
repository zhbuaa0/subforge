# subforge — 部署指南

subforge 支持三种部署形态：

| 形态 | 用途 | 适用模型 |
|---|---|---|
| **Windows 本地** | 单机、开发调试、≤10min 音频 | FunASR（paraformer-zh） |
| **WSL2 + 本机 GPU** | 全部 subforge 后端 + Windows 工具链互联 | 全部（含 MOSS） |
| **Linux 服务器 + vLLM** | 长音频 24/7 服务、横向扩展 | MOSS（推荐生产部署） |

本指南按 **WSL2**（推荐家庭 / 个人开发者）和 **Linux 生产**两条路径展开。
vLLM 在 Windows 上**无官方支持**（wheel 只有 manylinux），必须走 WSL2 或原生 Linux。

---

## 目录

1. [WSL2 部署（推荐 Windows 用户）](#1-wsl2-部署推荐-windows-用户)
2. [Linux 服务器原生部署](#2-linux-服务器原生部署)
3. [MOSS 加速方案（vLLM / sdpa）](#3-moss-加速方案vllm--sdpa)
   - [3.4 subforge 接入 vLLM](#34-subforge-接入-vllm)
4. [验证清单](#4-验证清单)
5. [运维](#5-运维)

---

## 1. WSL2 部署（推荐 Windows 用户）

### 为什么走 WSL2

- ✅ GPU 直通：WSL2 自带 NVIDIA 驱动，PyTorch 直接用
- ✅ vLLM 支持 only on Linux — WSL2 是 Windows 上唯一可行路径
- ✅ Linux 容器、systemd 都齐了
- ✅ 文件系统和 Windows 互通，`\\wsl$\Ubuntu\...` 直接访

### 1.1 准备 WSL2（一次性）

以管理员身份运行 **PowerShell**：

```powershell
# 安装 WSL2 + Ubuntu LTS（如果还没装）
wsl --install --no-distribution
wsl --set-default-version 2

# 如果已有 WSL 但版本是 1：
# wsl --set-version Ubuntu 2

# 进入 WSL
wsl
```

### 1.2 安装 NVIDIA WSL2 驱动（一次性）

在 **Windows** 上装：从 https://www.nvidia.com/Download/index.aspx 选 **Windows 64-bit** 的 **NVIDIA Studio / Game Ready** 驱动（≥ 472.50 含 WSL2 GPU 支持）。

确认在 WSL 里能看到 GPU：

```bash
nvidia-smi
```

输出应列出 NVIDIA GPU。如果空白，说明 Windows 显卡驱动没装或太旧。

### 1.3 WSL 里装系统依赖

```bash
sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv python3-pip \
    git curl wget \
    ffmpeg \
    build-essential ninja-build  # 编译 torch CUDA ext 时备用
```

> 如果 Ubuntu 默认 Python 是 3.12，请装 `python3.10 + venv`（subforge 要求 ≥ 3.10，可以更高）。

### 1.4 克隆代码

```bash
# WSL 文件系统和 Windows 互通；克隆到 /home/$USER 即可
git clone git@github.com:zhbuaa0/subforge.git
cd subforge
```

如果你在 Windows 上之前已经 git push 过，WSL 这边直接 `git clone` 即可，无需新密钥。

### 1.5 创建虚拟环境

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip uv
```

### 1.6 安装 PyTorch（按 CUDA 版本）

```bash
# 查看 CUDA 版本
nvidia-smi | head -8   # 右上角的 "CUDA Version: 12.x"

# CUDA 12.1（已知 RTX 4070 Ti / 30/40 系最稳的版本）
pip install torch>=2.8,<3 torchaudio>=2.8,<3 \
    --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4（新驱动）
pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124
```

> 想看具体能装的版本：`pip install torch== --dry-run --index-url https://download.pytorch.org/whl/cu121`

### 1.7 安装 subforge

```bash
# 装 + FunASR + MOSS 全部后端
pip install -e ".[funasr-runtime,moss-runtime,dev]"

# 仅 MOSS（GPU 推理长音频）
pip install -e ".[moss-runtime]"
```

### 1.8 测试 GPU

```bash
python -c "
import torch
print(f'torch:  {torch.__version__}')
print(f'cuda:   {torch.version.cuda}')
print(f'device: {torch.cuda.get_device_name(0)}')
print(f'vram:   {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
x = torch.randn(1024, 1024, device='cuda')
print(f'cuda matmul: {(x @ x.T).sum().item():.2f}')
"
```

预期输出：

```
torch:  2.x.x
cuda:   12.1
device: NVIDIA GeForce RTX 4070 Ti
vram:   12 GB
cuda matmul: -xxx.xx
```

### 1.9 启动服务

```bash
# 默认端口 8000
asr server --host 0.0.0.0 --port 8000

# 或后台 + 日志
nohup asr server --host 0.0.0.0 --port 8000 > subforge.log 2>&1 &
```

测试联通：

```bash
curl http://localhost:8000/health
# → {"status":"ok", ...}

curl http://localhost:8000/models | python -m json.tool | head -20
```

### 1.10 Windows 访问 WSL 里的服务

WSL 服务默认监听 `0.0.0.0:8000`。在 Windows 浏览器 / curl 直接访问：

```
http://localhost:8000/         # Web UI
http://localhost:8000/docs      # FastAPI 自动生成的 OpenAPI 文档
http://localhost:8000/health    # 健康检查
```

Windows 通过 WSL 的 `wsl-vpnkit` / `wslhost` 自动把 WSL IP 映射到 localhost，无需额外端口转发。

> 如果访问不到，在 WSL 里跑 `ip addr show eth0` 看 inet，把那个 IP 加进 Windows `C:\Windows\System32\drivers\etc\hosts`。

### 1.11 WSL 子模块：vLLM（可选）

如果需要更快的 MOSS 推理：

```bash
# === 必须用 uv pip，vLLM nightly 只在 wheels.vllm.ai ===
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129

# 测试
python -c "import vllm; print('vllm', vllm.__version__)"
```

启动 vLLM 服务（独立终端）：

```bash
vllm serve OpenMOSS-Team/MOSS-Transcribe-Diarize \
    --trust-remote-code \
    --port 8001
```

此时 subforge 跑在 8000，vLLM 跑在 8001。subforge v0.2+ 内置 `backend: vllm`
适配器（`subforge/backends/vllm_backend.py`），可以直接把推理转发到 vLLM，
详见 [§ 3.4](#34-subforge-接入-vllm)。

---

## 2. Linux 服务器原生部署

适用：Ubuntu 20.04+ / Debian 11+ / CentOS 8（用 dnf + EPEL） / Amazon Linux 2+。

### 2.1 系统依赖

Ubuntu / Debian：

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip \
    git curl wget ffmpeg build-essential ninja-build \
    libsox-dev  # mosstools / torchaudio 备用
```

CentOS / RHEL / Amazon Linux：

```bash
sudo dnf install -y python3.10 python3.10-devel \
    git curl wget ffmpeg gcc gcc-c++ make \
    sox-devel
```

### 2.2 创建用户和虚拟环境

生产不要 root 跑：

```bash
sudo useradd -m -s /bin/bash asr
sudo -u asr -i

git clone git@github.com:zhbuaa0/subforge.git
cd subforge
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
```

### 2.3 装 PyTorch + subforge

```bash
# 看 GPU + CUDA 版本
nvidia-smi

# 按你看到的 CUDA 选；RTX 40 系配 12.4 Pytorch wheel
pip install torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

pip install -e ".[funasr-runtime,moss-runtime,vllm-runtime]"
```

### 2.4 systemd 服务

创建 `/etc/systemd/system/subforge.service`：

```ini
[Unit]
Description=subforge ASR server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=asr
Group=asr
WorkingDirectory=/home/asr/subforge
Environment="PATH=/home/asr/subforge/.venv/bin"
ExecStart=/home/asr/subforge/.venv/bin/asr server --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
StandardOutput=append:/var/log/subforge/subforge.log
StandardError=append:/var/log/subforge/subforge.err
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

启用：

```bash
sudo mkdir -p /var/log/subforge && sudo chown asr:asr /var/log/subforge
sudo systemctl daemon-reload
sudo systemctl enable --now subforge
sudo systemctl status subforge   # 应显示 active (running)
sudo journalctl -u subforge -f    # 跟日志
```

### 2.5 Nginx 反向代理 + HTTPS（可选）

`/etc/nginx/conf.d/subforge.conf`：

```nginx
server {
    listen 80;
    server_name asr.your-domain.com;

    client_max_body_size 500M;   # 上传大音频

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_http_version 1.1;
        # SSE 长连接需要禁用缓冲
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }
}
```

HTTPS（Let's Encrypt）：

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d asr.your-domain.com
```

### 2.6 防火墙

```bash
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 8000/tcp  # subforge
sudo ufw allow 80,443/tcp # nginx (可选)
sudo ufw enable
```

---

## 3. MOSS 加速方案（vLLM / sdpa）

subforge 支持 FunASR 后端（Paraformer 系列）和 MOSS 后端。MOSS 在默认 HF transformers 下，长音频 RTF 退化严重。两个加速方案：

### 3.1 sdpa（零安装，推荐起步）

PyTorch 自带 scaled dot-product attention，给 MossBackend 加一行就能提速 1.5-2×。

编辑 `subforge/backends/moss_backend.py`，在 `from_pretrained` 加 `attn_implementation="sdpa"`：

```python
self._model = AutoModelForCausalLM.from_pretrained(
    self.spec.model,
    trust_remote_code=True,
    dtype=dtype,
    device_map="cuda:0",
    attn_implementation="sdpa",  # ← 加这行
)
```

不需要额外 pip 包。

### 3.2 Flash Attention 2（再快 1.3×）

```bash
pip install flash-attn --no-build-isolation
```

```python
attn_implementation="flash_attention_2",
```

### 3.3 vLLM 服务（生产部署，5-10×）

```bash
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129

vllm serve OpenMOSS-Team/MOSS-Transcribe-Diarize \
    --trust-remote-code \
    --port 8001 \
    --max-model-len 65536
```

预计速度对比（RTX 4070 Ti 12GB，70 分钟音频）：

| 方案 | RTF | 总耗时 |
|---|---|---|
| HF 默认 | 4.0 | ~4.7 小时 |
| HF + sdpa | 2.5 | ~3 小时 |
| HF + flash-attn 2 | 2.0 | ~2.3 小时 |
| **vLLM** | **0.3-0.6** | **~25-40 分钟** |

### 3.4 subforge 接入 vLLM

vLLM 服务起来后，subforge 注册表里已经包含 `moss-transcribe-diarize-vllm`
条目（`models.yaml` 默认 `backend: vllm`）。`asr transcribe --model <name>`
会走 OpenAI 兼容 `/v1/chat/completions` 把音频以 `data:audio/<fmt>;base64,...`
内联进 chat messages（不需要 vLLM 访问本地文件系统），用 `stream=True`
拉 token 增量同时驱动 stderr 进度条和 SSE 回调。

```bash
# 1. 装依赖（urllib 已经在 stdlib，vllm-runtime 只额外拉了 httpx 等)
pip install -e ".[vllm-runtime]"

# 2. 确认注册表项出现
asr models | grep moss-transcribe-diarize-vllm

# 3. 转写 — 让 asr 把请求转发给 vLLM
asr transcribe meeting.wav \
  --model moss-transcribe-diarize-vllm \
  --max-new-tokens 65536 \
  --format srt,json -o output/
```

#### 切换 vLLM 服务端

两种方式，等价：

```bash
# A. 设环境变量（影响 models.yaml 里 ${SUBFORGE_VLLM_BASE_URL:-...} 占位符）
export SUBFORGE_VLLM_BASE_URL=http://gpu-node-3:8001
asr transcribe meeting.wav --model moss-transcribe-diarize-vllm ...

# B. CLI 临时覆盖（不入环境变量)
asr transcribe meeting.wav \
  --model moss-transcribe-diarize-vllm \
  --vllm-url http://gpu-node-3:8001 \
  --format srt -o output/
```

#### 鉴权

vLLM 启用 API key 时：

```bash
# SUBFORGE_VLLM_API_KEY_ENV 是 subforge 要读的环境变量名（指明"在哪找 key"）
# VLLM_API_KEY 是真正的 key（vllm 服务端和 subforge 都用)
export SUBFORGE_VLLM_API_KEY_ENV=VLLM_API_KEY
export VLLM_API_KEY=sk-your-key
asr transcribe ... --model moss-transcribe-diarize-vllm ...
```

空 `init.api_key_env`（默认）= 无 `Authorization` 头，对应未鉴权的 vLLM。

#### 为什么 vLLM 后端无 weight，但 `backend: vllm` 仍出现在 `asr models`

`asr models` 默认会懒加载默认模型（warm），但 vLLM 后端构造时**不发起
网络请求** — 只 cache 配置，下次请求时再连服务端。这让 `asr server`
启动不会因为 vLLM 暂时没起来就崩；首个 `/transcribe` 调用才会触发连接
并按上述流程报错（明确说"vLLM 服务端不可达"）。

---

## 4. 验证清单

部署完跑这几个，确认一切正常：

```bash
# === 1. 包导入 ===
python -c "import subforge; print(subforge.__version__)"  # 应输出 0.2.0

# === 2. CLI ===
asr models
# 应列出 5 个模型，含 moss-transcribe-diarize

# === 3. 服务 ===
curl http://localhost:8000/health
curl http://localhost:8000/models | python -m json.tool

# === 4. 端到端转写 ===
# 用一个 30 秒以上的 wav 测试（不要太大，否则调试慢）
asr transcribe test.wav --model paraformer-zh -o /tmp/out -f srt
cat /tmp/out/test.srt
ls /tmp/out/   # 应有 test.srt / test.txt / test.json

# === 5. MOSS 后端（GPU 跑）===
asr transcribe test.wav --model moss-transcribe-diarize \
    --max-new-tokens 4096 -o /tmp/moss_out -f srt
# 应看到 [MOSS] xxx/4096 tokens ... 进度条

# === 6. Web UI ===
# 浏览器打开 http://localhost:8000/
# 选模型、上传文件、看进度条
```

如果某一步失败，按对应章节查错：
- 包导入失败 → §1.7 装 PyTorch；或 §1.5 venv
- `asr` 找不到 → §1.7 `pip install -e .` 失败
- 端口占住 → `lsof -i :8000` (Linux) / `netstat -ano | grep 8000` (Windows)
- OOM → 选更小的模型，或加 `--max-new-tokens 2048`

---

## 5. 运维

### 5.1 日志

```bash
# systemd
sudo journalctl -u subforge -f

# 手动启动
tail -F subforge.log
```

日志里看到的关键阶段：

| 日志 | 含义 |
|---|---|
| `loading model 'X' -> ...` | 正在加载模型 ID |
| `model ready in X.Xs` | 加载完成（首次 FunASR ~10s, MOSS ~1.5s） |
| `inference done in X.Xs` | 推理完成 |
| `[MOSS] X/65536 tokens ...` | MOSS 实时 token 进度 |
| `WARNING trust_remote_code: False` | FunASR 默认；MOSS 必须 True |
| `OOM` / `CUDA out of memory` | 显存不够；切模型或减小 max_new_tokens |

### 5.2 模型缓存清理

```bash
# FunASR（modelscope 缓存）
ls ~/.cache/modelscope/  # 几 GB
du -sh ~/.cache/modelscope/  # 占用查看
rm -rf ~/.cache/modelscope/models/iic/*   # 删某个子模型重新下载

# HuggingFace（MOSS）
ls ~/.cache/huggingface/hub/   # 模型权重
rm -rf ~/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Transcribe-Diarize
```

### 5.3 升级 subforge

```bash
cd /path/to/subforge
git pull
source .venv/bin/activate
pip install -e ".[funasr-runtime,moss-runtime]"
sudo systemctl restart subforge
```

### 5.4 升级依赖 PyTorch

PyTorch 升级需要匹配 CUDA 驱动版本：

```bash
nvidia-smi   # 看右上角 "CUDA Version"
# 然后到 https://pytorch.org/get-started/locally/ 选对应 CUDA wheel
pip install --upgrade torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu124  # 选你的 CUDA
```

### 5.5 性能监控

```bash
# GPU 实时
watch -n 1 nvidia-smi

# RTF（实时因子）
asr transcribe ... 2>&1 | grep "inference done"
# inference done in 42.6s for 5.5min audio => RTF = 42.6/327 = 0.13
```

RTF 越低越好：
- FunASR Paraformer：RTF ~0.1-0.2 ✅
- MOSS HF：RTF 1.0-4.0 (长音频退化)
- MOSS vLLM：RTF 0.3-0.6 ✅

---

## 常见问题 (FAQ)

**Q: `funasr` 装不上，错误 `Microsoft Visual C++ 14.0 required`**
A: 这是 Windows 包；装 conda 或用 WSL2。

**Q: `torch.cuda.is_available()` 返回 False（WSL）**
A: 三种可能：(1) Windows 显卡驱动太旧，升级到含 WSL2 GPU 支持的版本；(2) `nvidia-smi` 不输出，回到第 1.2 节重装驱动；(3) WSL kernel 太老，`wsl --update` 重启。

**Q: MOSS 显存爆炸 `CUDA OOM`**
A: MOSS 默认 max_new_tokens=65536 但实际可能不需要这么多；调小：
```bash
asr transcribe x.wav --model moss-transcribe-diarize --max-new-tokens 8192
```

**Q: `git push` 仍然 SSH 失败**
A: 把 `~/.ssh/config` 里所有 `140.82.112.4` 直 IP 配置删掉，让 SSH 走标准 DNS：
```bash
ssh-keygen -R 140.82.112.4
ssh-keygen -R github.com
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

**Q: WSL 里看不到 Windows D 盘**
A: 默认挂载 `/mnt/d` `/mnt/c`。要直接访问 subforge 项目在 `/mnt/d/.../subforge` 也行，但 I/O 慢。建议把代码放 WSL 文件系统里 (`/home/$USER/subforge`)。

---

更多细节见：[README.md](../README.md) · [DEVELOPMENT.md](DEVELOPMENT.md)
