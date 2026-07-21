# subforge — Python requirements
#
# 拆分成多份，便于在不同后端选装：
#
#   base.txt        # 核心运行时（纯 Python）
#   funasr.txt      # FunASR 后端（Paraformer / SenseVoice / 流式）
#   moss.txt        # MOSS 后端（HuggingFace transformers）
#   vllm.txt        # vLLM 客户端（OpenAI 协议调 vllm 服务端）
#   dev.txt         # 开发依赖（含 base）
#
# === 快速安装（CPU 模式，不需要 GPU）===
#   pip install -r requirements/base.txt -r requirements/funasr.txt -r requirements/moss.txt
#
# === GPU 模式（先装 torch）===
#   pip install torch>=2.8,<3 torchaudio>=2.8,<3 \
#     --index-url https://download.pytorch.org/whl/cu121
#   pip install -r requirements/base.txt -r requirements/funasr.txt -r requirements/moss.txt
#
# === 仅安装 subforge 包（+ 单个后端）===
#   pip install -e .                                   # 仅 base runtime
#   pip install -e .[moss-runtime]                      # 加 MOSS
#   pip install -e .[funasr-runtime]                    # 加 FunASR
#   pip install -e .[cpu-all]                           # 全部 CPU
#   pip install -e .[gpu-cu121]                         # 全部 GPU (CUDA 12.1)
#
# === 验证 ===
#   python -c "import subforge; print(subforge.__version__)"
#   python -m subforge.cli models

# NOTE: torch / torchaudio 不在这里约束，因为 CUDA wheel 必须按 CUDA 版本 + Python
# 版本严格匹配。请按 platforms/CUDA 从 https://pytorch.org/get-started/locally/ 选。
