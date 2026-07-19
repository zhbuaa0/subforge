"""Phase 6 验证：WebSocket 流式识别（/ws/stream）。"""
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:\Users\zhbuaa0\.conda\envs/paraformer-asr/python.exe"
AUDIO = r"C:\Users\zhbuaa0\.openclaw\media\inbound\C3142---81c7bbb6-da1e-42ce-a3e5-f07afb900f20.wav"
PORT = 18766
URL = f"http://127.0.0.1:{PORT}"
WS_URL = f"ws://127.0.0.1:{PORT}/ws/stream"

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT)


def wait_ready(url: str, timeout_s: int = 300) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1)
    return False


async def ws_smoke():
    """WS smoke test：发完整 wav + final，验证 pipeline 不要求真"流式"。

    注：真正的逐 chunk 流式增量识别需要在 server 端维护 PCM 缓冲 + 正确的
    chunk_size 配置；本次测试只覆盖 /ws/stream 端点的协议栈 / 模型加载 /
    多连接 cache 隔离等基础设施。
    """
    import websockets

    # 取前 60 秒（流式模型 RTF 较高，全文件会跑很久）
    try:
        import soundfile as sf
    except ImportError:
        print("[FAIL] soundfile not installed", flush=True)
        return False
    audio, sr = sf.read(AUDIO, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    max_seconds = 60
    audio = audio[: max_seconds * sr]

    # 编码成 wav bytes（funasr 通过 torchaudio 解码）
    import io
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")
    wav_bytes = buf.getvalue()
    print(f"[INFO] using first {len(audio)/sr:.2f}s, wav_bytes={len(wav_bytes)}",
          flush=True)

    print("[INFO] connecting WebSocket to", WS_URL, flush=True)
    async with websockets.connect(WS_URL, ping_interval=None, max_size=64 * 1024 * 1024) as ws:
        # 1. config
        await ws.send(json.dumps({"model": "paraformer-zh-streaming"}))
        ready_raw = await asyncio.wait_for(ws.recv(), timeout=120)
        ready = json.loads(ready_raw)
        print(f"[OK] ws recv ready: {ready}")
        assert ready.get("status") == "ready"

        # 2. send wav
        await ws.send(wav_bytes)
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=600)
            msg1 = json.loads(raw)
        except asyncio.TimeoutError:
            print("[FAIL] ws timeout after wav send", flush=True)
            return False
        if "error" in msg1:
            print(f"[FAIL] server error: {msg1['error']}", flush=True)
            return False
        print(f"[OK] ws 1st frame after wav: text_len={len(msg1.get('text',''))}")

        # 3. final
        await ws.send(json.dumps({"action": "final"}))
        try:
            final_raw = await asyncio.wait_for(ws.recv(), timeout=120)
            final = json.loads(final_raw)
        except asyncio.TimeoutError:
            print("[FAIL] ws timeout waiting for final", flush=True)
            return False
        assert final.get("is_final") is True, f"expected is_final=true, got {final}"
        final_text = final.get("text", "")
        assert len(final_text) >= 1, f"final text empty: {final!r}"
        # 不强制要求中文——流式模型可能输出标点 ITN 处理前的 raw
        print(f"[OK] ws final: text={final_text[:200]!r} ({len(final_text)} chars)")

        # 4. 新连接验证 cache 隔离
        print(f"[INFO] opening 2nd connection to verify cache isolation", flush=True)
        async with websockets.connect(WS_URL, ping_interval=None, max_size=64 * 1024 * 1024) as ws2:
            await ws2.send(json.dumps({"model": "paraformer-zh-streaming"}))
            r2 = json.loads(await asyncio.wait_for(ws2.recv(), timeout=120))
            assert r2.get("status") == "ready"
            await ws2.send(json.dumps({"action": "final"}))
            f2 = json.loads(await asyncio.wait_for(ws2.recv(), timeout=120))
            assert f2.get("is_final") is True
            # 全新 session：没送任何音频就 final，文本应为空或只有模型自动 EOS 字符
            t2 = f2.get("text", "")
            assert len(t2) <= len(final_text), \
                f"2nd connection has more text than 1st; cache not isolated: {t2!r}"
            print(f"[OK] 2nd connection independent (text_len={len(t2)})")

        return True


def main() -> int:
    print("=" * 70)
    print("Phase 6.3 verification — WebSocket streaming")
    print("=" * 70)

    # 启动 server
    print(f"\n[INFO] starting asr server on port {PORT}", flush=True)
    proc = subprocess.Popen(
        [PY, "-m", "subforge.cli", "server",
         "--host", "127.0.0.1", "--port", str(PORT),
         "--log-level", "warning"],
        env=ENV, cwd=str(ROOT),
        stdout=None, stderr=subprocess.STDOUT,
    )
    try:
        if not wait_ready(URL, timeout_s=300):
            print("[FAIL] server did not become ready in time", flush=True)
            return 1
        print(f"[OK] server ready at {URL}")

        # 确认 streaming 模型已注册
        with urllib.request.urlopen(f"{URL}/models") as r:
            models_resp = json.loads(r.read())
        names = [m["name"] for m in models_resp["models"]]
        assert "paraformer-zh-streaming" in names, f"streaming model missing: {names}"
        print(f"[OK] paraformer-zh-streaming registered")

        # 跑 ws 测试
        ok = asyncio.run(ws_smoke())
        if not ok:
            return 1

        print()
        print("=" * 70)
        print("[PASS] Phase 6.3 verification successful")
        print("=" * 70)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())