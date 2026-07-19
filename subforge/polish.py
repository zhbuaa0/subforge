"""AI 润色 —— 用 LLM 把 ASR 原始文本"轻度清理"。

设计原则：
- 用 OpenAI 兼容协议（OpenAI / DeepSeek / MiniMax / Qwen 等都能用）
- 默认"轻量"：删口头禅、修 ASR 错字、统一标点
- 保留 segments 的时间戳不变，只改 text
- 一次 API 调用处理一批段（默认 20 段/批），避免 190 段 = 190 次请求
- key 从环境变量读，不入 git

公开 API：
    load_polish_config(path=None) -> tuple[dict[str, ProviderSpec], str]
    polish_segments(segments, provider=...) -> list[Segment]
    polish_transcript(result, provider=...) -> TranscriptResult
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .transcriber import Segment, TranscriptResult

# 默认 models.yaml
_DEFAULT_YAML = Path(__file__).resolve().parent.parent / "models.yaml"


@dataclass
class ProviderSpec:
    """一个 LLM 润色 provider 的全部声明。

    api_key_env：环境变量名（不入 git）
    base_url：OpenAI 兼容 endpoint
    model：模型 ID
    temperature / max_tokens：默认参数
    batch_size：每次 API 调用合并多少段
    """

    name: str
    api_key_env: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096
    batch_size: int = 20
    timeout: float = 60.0
    extra_headers: dict[str, str] = field(default_factory=dict)


# ---------- 加载配置 ----------

def load_polish_config(path: str | Path | None = None) -> tuple[dict[str, ProviderSpec], str]:
    """从 models.yaml 的 `ai_polish` 段加载 provider 列表 + 默认 provider 名。"""
    p = Path(path) if path is not None else _DEFAULT_YAML
    if not p.exists():
        raise FileNotFoundError(f"models.yaml not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    section = raw.get("ai_polish") or {}
    if not section:
        raise ValueError(f"{p}: 缺少 ai_polish 段")

    providers_raw = section.get("providers") or {}
    if not providers_raw:
        raise ValueError(f"{p}: ai_polish.providers 为空")

    out: dict[str, ProviderSpec] = {}
    for name, body in providers_raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"provider '{name}': 必须是 mapping")
        for required in ("api_key_env", "base_url", "model"):
            if required not in body:
                raise ValueError(f"provider '{name}': 缺少 '{required}'")
        out[name] = ProviderSpec(
            name=name,
            api_key_env=str(body["api_key_env"]),
            base_url=str(body["base_url"]).rstrip("/"),
            model=str(body["model"]),
            temperature=float(body.get("temperature", 0.2)),
            max_tokens=int(body.get("max_tokens", 4096)),
            batch_size=int(body.get("batch_size", 20)),
            timeout=float(body.get("timeout", 60.0)),
            extra_headers=dict(body.get("extra_headers") or {}),
        )

    default_name = str(section.get("default") or next(iter(out)))
    if default_name not in out:
        raise ValueError(f"{p}: ai_polish.default='{default_name}' 不在 providers 中")
    return out, default_name


def get_api_key(spec: ProviderSpec) -> str:
    """从环境变量取 key；找不到给清晰报错。"""
    val = os.environ.get(spec.api_key_env)
    if not val:
        raise RuntimeError(
            f"provider '{spec.name}' 需要环境变量 {spec.api_key_env} 提供 API key。"
            f"设置：$env:{spec.api_key_env} = \"sk-...\""
        )
    return val


# ---------- Prompt ----------

_SYSTEM_PROMPT_LIGHT = """你是一个 ASR 转写文本的"轻度清理"助手。

任务：保持原意不变，只做以下清理：
1. 删除无意义口头禅：嗯 / 啊 / 呃 / 那个 / 这个 / 然后 / 对 / 哎 / 嗨 / 哦 等（句首句末单独存在的）
2. 修正明显 ASR 错字（同音字、近音字、专业术语），但不要改写用户原话
3. 统一标点符号（中文段落用全角，英文/数字周围用半角）
4. 保留所有时间戳对应的内容；只改 text 字段

输出：JSON 数组，元素是清理后的字符串数组，**与输入严格一一对应**（数量、顺序都不能变）。
不要改写句子结构，不要添加内容，不要做摘要或翻译。"""


def _user_prompt_batch(items: list[dict[str, Any]]) -> str:
    """把一批 segment 编成 user prompt。"""
    payload = [{"i": i, "t": s["text"]} for i, s in enumerate(items)]
    return (
        "下面是 ASR 转写片段，请按系统指令轻度清理，输出 JSON 数组：\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=1)}\n```\n\n"
        "返回（仅 JSON，不要解释）：\n```json\n"
    )


def _parse_json_array(content: str, expected_len: int) -> list[str] | None:
    """从 LLM 输出里抠出 JSON 数组。容忍 ```json ... ``` 包裹。"""
    s = content.strip()
    # 去掉首尾 code fence
    if s.startswith("```"):
        # 去掉首行 ```json / ```
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    # 找 [ ... ]
    lb, rb = s.find("["), s.rfind("]")
    if lb == -1 or rb == -1 or rb <= lb:
        return None
    try:
        arr = json.loads(s[lb:rb + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list) or len(arr) != expected_len:
        return None
    out: list[str] = []
    for x in arr:
        if not isinstance(x, str):
            return None
        out.append(x)
    return out


# ---------- HTTP 调用（标准库 urllib，无 SDK 依赖） ----------

def _http_post_json(url: str, headers: dict[str, str], body: dict, timeout: float) -> dict:
    import urllib.request

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _call_openai_compat(spec: ProviderSpec, system: str, user: str) -> str:
    """调一次 OpenAI Chat Completions 兼容接口，返回 content 字符串。"""
    key = get_api_key(spec)
    body = {
        "model": spec.model,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {key}",
        **spec.extra_headers,
    }
    resp = _http_post_json(
        f"{spec.base_url}/chat/completions",
        headers=headers,
        body=body,
        timeout=spec.timeout,
    )
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"LLM 返回结构异常: {resp!r}") from e


# ---------- 主流程 ----------

def _polish_batch(items: list[dict[str, Any]], spec: ProviderSpec) -> list[str]:
    """对一批段做润色。失败时返回原文（不抛异常，让用户能看到结果）。"""
    if not items:
        return []
    user = _user_prompt_batch(items)
    last_err: Exception | None = None
    for attempt in range(2):  # 最多重试 1 次
        try:
            content = _call_openai_compat(spec, _SYSTEM_PROMPT_LIGHT, user)
            parsed = _parse_json_array(content, len(items))
            if parsed is not None:
                return parsed
            last_err = ValueError(f"无法解析 LLM 输出为长度 {len(items)} 的数组: {content[:200]!r}")
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    # 失败 fallback：返回原文
    print(f"[WARN] polish 批次失败，回退原文: {last_err}", flush=True)
    return [s["text"] for s in items]


def polish_segments(
    segments: list[Segment],
    *,
    spec: ProviderSpec,
) -> list[Segment]:
    """对 Segment 列表做润色，返回新的 Segment 列表（时间戳不变）。"""
    if not segments:
        return []
    out: list[Segment] = []
    for start in range(0, len(segments), spec.batch_size):
        batch = segments[start:start + spec.batch_size]
        items = [{"i": i, "text": s.text} for i, s in enumerate(batch)]
        polished_texts = _polish_batch(items, spec)
        for seg, new_text in zip(batch, polished_texts):
            out.append(replace(seg, text=new_text))
        # 进度提示
        done = min(start + spec.batch_size, len(segments))
        print(f"[INFO] polish: {done}/{len(segments)} segments", flush=True)
    return out


def polish_transcript(
    result: TranscriptResult,
    *,
    spec: ProviderSpec,
) -> TranscriptResult:
    """对 TranscriptResult 做润色，返回新 TranscriptResult（保留 text/segments/num_speakers）。"""
    new_segments = polish_segments(result.segments, spec=spec)
    new_text = "".join(s.text for s in new_segments)
    return TranscriptResult(
        text=new_text,
        segments=new_segments,
        num_speakers=result.num_speakers,
        language=result.language,
        raw=result.raw,
    )


# ---------- Dry-run（key 缺失时的占位实现） ----------

def is_provider_available(spec: ProviderSpec) -> bool:
    """检查 provider 是否可用（环境变量是否设置），不实际发请求。"""
    return bool(os.environ.get(spec.api_key_env))


__all__ = [
    "ProviderSpec",
    "load_polish_config",
    "get_api_key",
    "is_provider_available",
    "polish_segments",
    "polish_transcript",
]