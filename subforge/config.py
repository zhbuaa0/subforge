"""models.yaml 加载与校验。

唯一负责把 YAML → `ModelSpec` dataclass 的地方；其他模块不接受裸 dict。

公开 API：
    load_registry(path: str | None = None) -> tuple[dict[str, ModelSpec], str]
        加载并校验 models.yaml，返回 (按 name 索引的 ModelSpec 字典, 默认模型 name)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import os
import re

# models.yaml 与本文件相对路径（仓库根）
_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "models.yaml"


# ---------------------------------------------------------------------------
# Lightweight YAML string expansion
# ---------------------------------------------------------------------------
# Supports the bash-style ``${VAR}`` and ``${VAR:-default}`` references in
# ``models.yaml`` so users can wire environment variables into the registry
# without a templating layer:

#     base_url: ${SUBFORGE_VLLM_BASE_URL:-http://127.0.0.1:8001}
#     api_key_env: ${SUBFORGE_VLLM_API_KEY_ENV:-}

# Rules:
#   * ``${VAR}``           -- substitute $VAR; raise if unset (we want loud
#                             failure for typos, not silent empty strings).
#   * ``${VAR:-default}``  -- substitute $VAR if set and non-empty, else
#                             ``default``. Empty $VAR also falls back.
#   * Anything else is left alone (literal ``$`` characters are uncommon
#     in URLs / model IDs / API key names so this stays simple).
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_str(s: str) -> str:
    def repl(m: re.Match[str]) -> str:
        var, default = m.group(1), m.group(2)
        val = os.environ.get(var)
        if val:
            return val
        if default is not None:
            return default
        raise ValueError(
            f"models.yaml: 环境变量 ${var} 未设且无默认值;"
            " 请在 shell 里 export 后重试"
        )

    return _ENV_REF_RE.sub(repl, s)


def _expand_env_in(obj: Any) -> Any:
    """Recursively expand env refs in nested dict / list / string leaves.

    Used by ``_coerce_spec`` to scrub ``init`` and ``generate`` blocks. Features
    stay as raw strings (treating ``${...}`` as opaque metadata) so the user
    can read them as-is in ``asr models`` output.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_in(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_in(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_str(obj)
    return obj


@dataclass
class ModelSpec:
    """一个 ASR 模型在注册表里的全部声明。

    只有 name/model 必填；其余子模型和参数按需设置，None/空 表示跳过。
    """

    name: str
    model: str
    backend: str = "funasr"  # "funasr" (default), "moss", or "vllm"
    vad_model: str | None = None
    punc_model: str | None = None
    spk_model: str | None = None
    init: dict[str, Any] = field(default_factory=dict)
    generate: dict[str, Any] = field(default_factory=dict)
    postprocess: str | None = None
    streaming: bool = False
    features: dict[str, Any] = field(default_factory=dict)

    # ---- 衍生便利属性 ----

    @property
    def has_vad(self) -> bool:
        return bool(self.vad_model)

    @property
    def has_punc(self) -> bool:
        return bool(self.punc_model)

    @property
    def has_spk(self) -> bool:
        return bool(self.spk_model)

    def auto_model_kwargs(self) -> dict[str, Any]:
        """构造 funasr.AutoModel(...) 时使用的 kwargs。

        只传设了值的子模型；None 字段不传，让 FunASR 按模型自带能力走。
        """
        kw: dict[str, Any] = {"model": self.model, **self.init}
        if self.vad_model is not None:
            kw["vad_model"] = self.vad_model
        if self.punc_model is not None:
            kw["punc_model"] = self.punc_model
        if self.spk_model is not None:
            kw["spk_model"] = self.spk_model
        return kw


def _coerce_spec(name: str, raw: dict[str, Any]) -> ModelSpec:
    if not isinstance(raw, dict):
        raise ValueError(f"model '{name}': 必须是 mapping，实际为 {type(raw).__name__}")
    if "model" not in raw:
        raise ValueError(f"model '{name}': 缺少必填字段 'model'")

    backend = str(raw.get("backend", "funasr"))
    if backend not in {"funasr", "moss", "vllm"}:
        raise ValueError(
            f"model '{name}': backend='{backend}' 不支持；可选：funasr, moss, vllm"
        )

    streaming = bool(raw.get("streaming", False))
    features = dict(raw.get("features") or {})
    # streaming 字段自动同步到 features，features 里再覆盖
    features.setdefault("streaming", streaming)
    features.setdefault("backend", backend)

    return ModelSpec(
        name=name,
        model=str(raw["model"]),
        backend=backend,
        vad_model=raw.get("vad_model"),
        punc_model=raw.get("punc_model"),
        spk_model=raw.get("spk_model"),
        # Expand ${VAR:-default} so users can point the vLLM backend at
        # whatever endpoint matches their environment.
        init=_expand_env_in(dict(raw.get("init") or {})),
        generate=_expand_env_in(dict(raw.get("generate") or {})),
        postprocess=raw.get("postprocess"),
        streaming=streaming,
        features=features,
    )


def load_registry(path: str | Path | None = None) -> tuple[dict[str, ModelSpec], str]:
    """加载并校验 models.yaml。

    Args:
        path: YAML 文件路径；None 时使用仓库根的 `models.yaml`。

    Returns:
        (specs_by_name, default_name)；
        specs_by_name 至少包含一个条目；default_name 必须是其中之一。

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式 / 字段 / default 引用不合法
    """
    p = Path(path) if path is not None else _DEFAULT_YAML
    if not p.exists():
        raise FileNotFoundError(f"models.yaml not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{p}: 顶层必须是 mapping，实际为 {type(raw).__name__}")

    raw_models = raw.get("models") or {}
    if not raw_models:
        raise ValueError(f"{p}: 'models' 段为空或缺失")
    if not isinstance(raw_models, dict):
        raise ValueError(f"{p}: 'models' 必须是 mapping")

    specs: dict[str, ModelSpec] = {}
    for name, body in raw_models.items():
        specs[str(name)] = _coerce_spec(str(name), body)

    default_name = str(raw.get("default") or next(iter(specs)))
    if default_name not in specs:
        raise ValueError(f"{p}: default='{default_name}' 不在 models 列表中")
    return specs, default_name