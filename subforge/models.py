"""Lazy-loaded model registry — backend-agnostic.

Returns a ``BaseBackend`` instance (``FunasrBackend`` or ``MossBackend``) keyed
by the name registered in ``models.yaml``.  Cached per-name; thread-safe.

The model registry is the single dispatch point that knows which backend
implements which model.  Downstream callers (``Transcriber``, ``cli.py``,
``server.py``) only see the ``BaseBackend`` contract.
"""

from __future__ import annotations

import threading
from typing import Any

from .config import ModelSpec


class ModelRegistry:
    """按 spec 构造并缓存 backend 实例。线程安全。

    用法：
        specs, default = load_registry()
        registry = ModelRegistry(specs)
        backend = registry.get("paraformer-zh")     # 首次构造，后续命中缓存
        backend2 = registry.get(default)            # 等同上

    Args:
        specs: 已校验的 ModelSpec 字典（来自 `config.load_registry`）
    """

    def __init__(self, specs: dict[str, ModelSpec]):
        if not specs:
            raise ValueError("ModelRegistry: 空 spec 字典")
        self._specs = dict(specs)
        self._cache: dict[str, Any] = {}  # name -> BaseBackend
        self._lock = threading.Lock()

    # ---- 查询 ----

    def names(self) -> list[str]:
        """所有可用模型名（按 specs 声明顺序）。"""
        return list(self._specs.keys())

    def spec(self, name: str) -> ModelSpec:
        if name not in self._specs:
            raise KeyError(f"未知模型 '{name}'；可用：{self.names()}")
        return self._specs[name]

    def features(self, name: str) -> dict[str, Any]:
        return dict(self.spec(name).features)

    # ---- 取 backend ----

    def get(self, name: str) -> Any:
        """懒加载并缓存 backend 实例。线程安全。

        Returns either a ``FunasrBackend`` or ``MossBackend`` depending on
        ``spec.backend``.  Callers should treat the result as a
        ``BaseBackend`` (duck-typed).
        """
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                return cached
            spec = self.spec(name)
            backend = self._build_backend(spec)
            self._cache[name] = backend
            return backend

    def get_many(self, names: list[str]) -> dict[str, Any]:
        """一次加载多个 backend（用于服务启动时预热默认 + 流式）。"""
        return {n: self.get(n) for n in names}

    def warm(self, names: list[str] | None = None) -> list[str]:
        """显式预热；返回成功加载的 name 列表。names=None 表示全部。"""
        targets = names if names is not None else self.names()
        return [n for n in targets if self.get(n) is not None]

    def is_cached(self, name: str) -> bool:
        return name in self._cache

    @staticmethod
    def _build_backend(spec: ModelSpec) -> Any:
        """Dispatch to the right backend class based on ``spec.backend``.

        Importing backends lazily keeps the FunASR-only code path from
        paying any import cost for MOSS / transformers.
        """
        if spec.backend == "moss":
            from .backends.moss_backend import MossBackend

            return MossBackend(spec)
        # default — preserves original funasr-only behaviour
        from .backends.funasr_backend import FunasrBackend

        return FunasrBackend(spec)