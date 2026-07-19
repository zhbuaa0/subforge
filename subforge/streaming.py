"""WebSocket 流式识别会话。

每个连接 = 一个 StreamingSession，独立的 cache dict（流式模型的状态）。
push(pcm_chunk, is_final) -> 返回增量文本（partial / final）。

注意：
- 必须用 `*-online` 模型 + `cache={}` 起手；切勿用批量模型
- 流式 Paraformer 没有说话人分离、无 sentence_info，只有逐 chunk 的 text
- 同步阻塞 generate() —— 调用方应在 threadpool 中跑
"""

from __future__ import annotations

import threading
from typing import Any

from .config import ModelSpec


class StreamingSession:
    """单连接流式会话。线程安全（同一会话不会被并发调用）。"""

    def __init__(self, model: Any, spec: ModelSpec):
        if not spec.streaming:
            raise ValueError(
                f"model '{spec.name}' 不是流式模型（spec.streaming=False）；"
                f"请用 -online 模型 + cache 参数"
            )
        self.model = model
        self.spec = spec
        self.cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        # 累计全文 + 历史已发文本（用于 diff）
        self._accumulated_text: str = ""
        self._sent_len: int = 0

    def push(self, audio, *, is_final: bool = False) -> dict[str, Any]:
        """送入一段音频（文件路径 / numpy 数组 / 字节），返回增量结果。

        Returns:
            {"partial": <str>, "is_final": <bool>, "text": <str>}
            - partial: 自上次发送以来的新增文本
            - text: 累计全文
        """
        with self._lock:
            kwargs = dict(self.spec.generate)
            kwargs["cache"] = self.cache
            kwargs["is_final"] = is_final
            raw = self.model.generate(input=audio, **kwargs)

        text = ""
        if raw and isinstance(raw, list) and raw and isinstance(raw[0], dict):
            text = str(raw[0].get("text", "") or "")
        elif isinstance(raw, dict):
            text = str(raw.get("text", "") or "")

        # FunASR 流式 model 的 text 是"到目前为止的累计文本"
        # 计算增量
        if text.startswith(self._accumulated_text):
            partial = text[len(self._accumulated_text):]
        else:
            # 罕见：模型重置 / cache 漂移；把全文当 partial
            partial = text
        self._accumulated_text = text

        return {
            "partial": partial,
            "text": text,
            "is_final": is_final,
        }

    def reset(self) -> None:
        """重置 cache 与累计文本（新一段对话复用同一 session 时用）。"""
        with self._lock:
            self.cache = {}
            self._accumulated_text = ""
            self._sent_len = 0