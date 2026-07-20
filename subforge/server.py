"""FastAPI HTTP 服务。

路由：
    GET  /models           列出已注册模型
    GET  /health           简单健康检查
    POST /transcribe       multipart 上传音频；返回 JSON + 可选格式内嵌

设计：
    - 模型在 lifespan 内一次性加载（默认模型 warm），跨请求复用
    - GPU 推理串行化：单 asyncio.Lock（单卡用户级服务足够）
    - 上传字节落到 tempdir，FunASR 通过路径读；如格式非 wav/flac 自动 ffmpeg 转码
    - 转码/推理/导出 全在 lifespan 启动的 threadpool 里跑（FastAPI sync 自动行为）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# UTF-8 stdout for clean logs
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import audio as audio_utils
from . import exporters
from . import polish as polish_mod
from .config import load_registry
from .models import ModelRegistry
from .streaming import StreamingSession
from .transcriber import Transcriber

WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"

logger = logging.getLogger("subforge.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    specs, default = load_registry()
    registry = ModelRegistry(specs)

    app.state.specs = specs
    app.state.default = default
    app.state.registry = registry
    app.state.gpu_lock = asyncio.Lock()

    # eager warm default model so first request is fast
    try:
        logger.info("warming default model '%s' ...", default)
        registry.get(default)
        logger.info("default model ready")
    except Exception as e:  # noqa: BLE001
        logger.warning("eager load failed: %s (will lazy-load on first request)", e)

    yield
    # cleanup: drop model refs
    app.state.registry._cache.clear()  # type: ignore[attr-defined]


app = FastAPI(
    title="paraformer-asr",
    description="FunASR / ModelScope 驱动的多模型中文语音识别 HTTP 服务",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------- 路由 ----------

@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_model": getattr(app.state, "default", None),
        "ffmpeg_available": audio_utils.check_ffmpeg(),
        "native_formats": sorted(audio_utils._NATIVE_FORMATS),
    }


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
async def web_ui() -> FileResponse:
    """内置 Web UI（上传 + 结果展示 + 下载）。"""
    if not WEB_INDEX.exists():
        raise HTTPException(500, f"web ui not found: {WEB_INDEX}")
    return FileResponse(WEB_INDEX, media_type="text/html; charset=utf-8")


@app.get("/models")
async def list_models() -> dict[str, Any]:
    specs: dict = app.state.specs
    default: str = app.state.default

    # polish providers
    try:
        p_providers, p_default = polish_mod.load_polish_config()
        polish_info = {
            "default": p_default,
            "providers": [
                {
                    "name": n,
                    "model": p.model,
                    "available": polish_mod.is_provider_available(p),
                    "api_key_env": p.api_key_env,
                }
                for n, p in p_providers.items()
            ],
        }
    except Exception as e:
        polish_info = {"error": str(e)}

    return {
        "default": default,
        "ffmpeg_available": audio_utils.check_ffmpeg(),
        "models": [
            {
                "name": name,
                "model_id": spec.model,
                "backend": spec.backend,
                "has_vad": spec.has_vad,
                "has_punc": spec.has_punc,
                "has_spk": spec.has_spk,
                "features": spec.features,
                "cached": app.state.registry.is_cached(name),
            }
            for name, spec in specs.items()
        ],
        "polish": polish_info,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(..., description="音频文件（wav/flac 免依赖；其他需 ffmpeg）"),
    model: str | None = Form(None, description="模型名（默认：注册表 default）"),
    formats: str | None = Form(None, description="可选：逗号分隔的内嵌格式，如 srt,vtt,txt"),
    preset_spk_num: int | None = Form(None, description="强制说话人数（仅支持 spk 的模型）"),
    language: str | None = Form(None, description="语言（多语种模型）"),
    polish: str | None = Form(None, description="可选：polish provider 名（minimax/deepseek/openai）"),
) -> JSONResponse:
    name = model or app.state.default
    specs = app.state.specs
    if name not in specs:
        raise HTTPException(404, f"unknown model '{name}'; available: {list(specs)}")
    spec = specs[name]

    if preset_spk_num is not None and not spec.has_spk:
        raise HTTPException(400, f"model '{name}' has no spk_model; preset_spk_num not applicable")
    if spec.streaming:
        raise HTTPException(
            400,
            f"model '{name}' is streaming-only; use /ws/stream instead of /transcribe",
        )

    content = await audio.read()
    if not content:
        raise HTTPException(400, "empty upload")

    suffix = audio_utils.guess_suffix(audio.filename, audio.content_type)
    tmp = audio_utils.save_upload(content, suffix)
    converted: Path | None = None
    try:
        converted = audio_utils.ensure_native_format(tmp)
    except RuntimeError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(415, str(e)) from e

    try:
        gpu_lock: asyncio.Lock = app.state.gpu_lock
        async with gpu_lock:
            transcriber = Transcriber(app.state.registry.get(name), spec)
            results = transcriber.transcribe(
                str(converted),
                preset_spk_num=preset_spk_num,
                **({"language": language} if language else {}),
            )
        r = results[0]

        response: dict[str, Any] = {
            "model": name,
            "text": r.text,
            "num_speakers": r.num_speakers,
            "language": r.language,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "spk": s.spk,
                }
                for s in r.segments
            ],
        }

        # 内嵌格式
        requested = [f.strip().lower() for f in (formats or "").split(",") if f.strip()]
        for fmt in requested:
            if fmt not in exporters.available_formats():
                raise HTTPException(400, f"unknown format '{fmt}'; available: {exporters.available_formats()}")
            response[fmt] = exporters.render(r, fmt)

        # 可选：AI 润色（用 LLM 轻度清理 segment.text）
        if polish:
            providers, default_name = polish_mod.load_polish_config()
            pname = polish.strip()
            if pname not in providers:
                raise HTTPException(400, f"unknown polish provider '{pname}'; available: {list(providers)}")
            pspec = providers[pname]
            if not polish_mod.is_provider_available(pspec):
                raise HTTPException(
                    503,
                    f"polish provider '{pname}' unavailable: env {pspec.api_key_env} not set on server",
                )
            try:
                polished = await asyncio.to_thread(polish_mod.polish_transcript, r, pspec)
            except Exception as e:
                raise HTTPException(500, f"polish failed: {e}") from e
            response["polished"] = {
                "provider": pname,
                "model": pspec.model,
                "text": polished.text,
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text, "spk": s.spk}
                    for s in polished.segments
                ],
            }
            # 同时也以 polished.<fmt> 输出
            for fmt in requested:
                response[f"polished_{fmt}"] = exporters.render(polished, fmt)

        return JSONResponse(response)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        if converted is not None and converted.exists() and converted != tmp:
            try:
                converted.unlink(missing_ok=True)
            except OSError:
                pass


# ---------- WebSocket 流式 ----------

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """WebSocket 流式识别。

    协议：
      1. 客户端连接
      2. 首条文本消息：{"model": "<name>"}     —— 指定流式模型
      3. 后续二进制消息：音频 chunk（PCM 16k mono f32 或 wav 文件）
      4. 文本消息 {"action": "final"}           —— 标记最后一块
      5. 服务端持续回文本消息 {"partial": "...", "text": "...", "is_final": false}
      6. 最终块后回 {"partial": "", "text": "<完整文本>", "is_final": true}

    注意：
      - 模型必须在 models.yaml 标记 streaming: true
      - 流式 Paraformer 不支持说话人分离
    """
    await ws.accept()

    model_name: str | None = None
    session: StreamingSession | None = None
    try:
        # 1. 等客户端配置消息
        first = await ws.receive_text()
        try:
            cfg = json.loads(first)
        except json.JSONDecodeError as e:
            await ws.send_json({"error": f"first message must be JSON config: {e}"})
            await ws.close(code=4400)
            return
        model_name = cfg.get("model") or app.state.default
        specs = app.state.specs
        if model_name not in specs:
            await ws.send_json({"error": f"unknown model '{model_name}'; available: {list(specs)}"})
            await ws.close(code=4404)
            return
        spec = specs[model_name]
        if not spec.streaming:
            await ws.send_json({
                "error": f"model '{model_name}' is not streaming; use /transcribe or a -online model"
            })
            await ws.close(code=4400)
            return

        # 加载流式模型（首次会下载 ~1GB）
        gpu_lock: asyncio.Lock = app.state.gpu_lock
        async with gpu_lock:
            streaming_model = app.state.registry.get(model_name)
        session = StreamingSession(streaming_model, spec)
        await ws.send_json({"status": "ready", "model": model_name})

        # 2. 主循环
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "text" in msg:
                # 控制消息
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"error": "non-JSON text frame not supported"})
                    continue
                action = payload.get("action")
                if action == "final":
                    # 空推一次以触发最终输出
                    async with gpu_lock:
                        result = await asyncio.to_thread(session.push, b"", is_final=True)
                    await ws.send_json(result)
                    break
                elif action == "reset":
                    session.reset()
                    await ws.send_json({"status": "reset"})
                else:
                    await ws.send_json({"error": f"unknown action: {action!r}"})

            elif "bytes" in msg:
                data: bytes = msg["bytes"]
                if not data:
                    continue
                # FunASR 流式模型不接受 raw bytes（不像 batch 那样有 torchaudio 兼容层）；
                # 把 bytes 当 wav 文件落盘后传给 generate()。
                tmp = audio_utils.save_upload(data, suffix=".wav")
                try:
                    async with gpu_lock:
                        result = await asyncio.to_thread(session.push, str(tmp), is_final=False)
                finally:
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass
                await ws.send_json(result)

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        logger.exception("ws_stream error: %s", e)
        try:
            await ws.send_json({"error": str(e)})
            await ws.close(code=4500)
        except Exception:
            pass
    finally:
        # 不清缓存：session 可能被复用（同一连接生命周期内）
        pass


# ---------- 入口 ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="paraformer-asr HTTP 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="开发模式（auto-reload）")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    try:
        import uvicorn  # noqa: F401
    except ImportError:
        print(
            "[ERR] 未安装 uvicorn；运行: pip install 'uvicorn[standard]'",
            file=sys.stderr,
        )
        return 2

    import uvicorn
    uvicorn.run(
        "subforge.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())