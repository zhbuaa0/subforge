"""AutoModel 实例化与缓存。

唯一调用 `funasr.AutoModel(...)` 的地方。其他模块需要模型时，通过 `ModelRegistry.get(name)` 获取。
服务模式下 registry 单例 + 缓存，避免每次请求重新加载模型（冷启动 ~10s 量级）。
"""

from __future__ import annotations

import threading
from typing import Any

from funasr import AutoModel

from .config import ModelSpec

# funasr.AutoModel 的真实类型；我们在静态类型里直接当 Any 用，避免被锁死的子类签名绑住
_FunasrModel = Any


class ModelRegistry:
    """按 spec 构造并缓存 AutoModel 实例。线程安全。

    用法：
        specs, default = load_registry()
        registry = ModelRegistry(specs)
        model = registry.get("paraformer-zh")     # 首次构造，后续命中缓存
        model2 = registry.get(default)             # 等同上

    Args:
        specs: 已校验的 ModelSpec 字典（来自 `config.load_registry`）
    """

    def __init__(self, specs: dict[str, ModelSpec]):
        if not specs:
            raise ValueError("ModelRegistry: 空 spec 字典")
        self._specs = dict(specs)
        self._cache: dict[str, _FunasrModel] = {}
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

    # ---- 取模型 ----

    def get(self, name: str) -> _FunasrModel:
        """懒加载并缓存 AutoModel 实例。线程安全。"""
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                return cached
            spec = self.spec(name)
            kwargs = spec.auto_model_kwargs()
            model = AutoModel(**kwargs)
            self._cache[name] = model
            return model

    def get_many(self, names: list[str]) -> dict[str, _FunasrModel]:
        """一次加载多个模型（用于服务启动时预热默认 + 流式）。"""
        return {n: self.get(n) for n in names}

    def warm(self, names: list[str] | None = None) -> list[str]:
        """显式预热；返回成功加载的 name 列表。names=None 表示全部。"""
        targets = names if names is not None else self.names()
        return [n for n in targets if self.get(n) is not None]

    def is_cached(self, name: str) -> bool:
        return name in self._cache