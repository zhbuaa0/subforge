"""Phase 5 验证：HTTP 服务 (FastAPI)。

起 asr server 后台进程 → 调 /health, /models, /transcribe → 校验响应。
"""
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
PY = r"C:\Users\zhbuaa0\.conda/envs/paraformer-asr/python.exe"
AUDIO = r"C:\Users\zhbuaa0\.openclaw\media\inbound\C3142---81c7bbb6-da1e-42ce-a3e5-f07afb900f20.wav"
PORT = 18765  # 避免与默认 8000 冲突
URL = f"http://127.0.0.1:{PORT}"

ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT)


def wait_ready(url: str, timeout_s: int = 300) -> bool:
    """轮询 /health 直到返回 200 或超时。"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1)
    return False


def main() -> int:
    print("=" * 70)
    print("Phase 5.3 verification — HTTP server")
    print("=" * 70)

    # ---------- 启动 server ----------
    print(f"\n[INFO] starting asr server on port {PORT}", flush=True)
    proc = subprocess.Popen(
        [PY, "-m", "subforge.cli", "server",
         "--host", "127.0.0.1", "--port", str(PORT),
         "--log-level", "warning"],
        env=ENV, cwd=str(ROOT),
        # 不 capture stdout/stderr，让 server 直接输出到我们的 stdout
        # 否则 PIPE buffer 满后会阻塞子进程
        stdout=None, stderr=subprocess.STDOUT,
    )
    try:
        if not wait_ready(URL, timeout_s=300):
            print("[FAIL] server did not become ready in time", flush=True)
            return 1
        print(f"[OK] server ready at {URL}")

        # ---------- 1. /health ----------
        with urllib.request.urlopen(f"{URL}/health") as r:
            health = json.loads(r.read())
        assert health["status"] == "ok"
        assert "default_model" in health
        print(f"[OK] /health: {health}")

        # ---------- 2. /models ----------
        with urllib.request.urlopen(f"{URL}/models") as r:
            models_resp = json.loads(r.read())
        names = [m["name"] for m in models_resp["models"]]
        assert "paraformer-zh" in names, f"missing paraformer-zh: {names}"
        assert "sensevoice" in names, f"missing sensevoice: {names}"
        print(f"[OK] /models: default={models_resp['default']}, models={names}")

        # ---------- 3. /transcribe (paraformer-zh) ----------
        # 用 requests 上传文件（urllib 也行但 requests 更直观；这里用 stdlib + multipart）
        import http.client
        boundary = "----ParaformerTestBoundary12345"
        with open(AUDIO, "rb") as f:
            audio_bytes = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="C3142.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + audio_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
        conn.request(
            "POST", "/transcribe",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 200, f"transcribe failed: {resp.status} {raw[:500]!r}"
        result = json.loads(raw)
        conn.close()

        print(f"[OK] /transcribe (paraformer-zh): status=200, "
              f"segments={len(result['segments'])}, "
              f"num_speakers={result['num_speakers']}, "
              f"text_len={len(result['text'])}")
        assert result["model"] == "paraformer-zh"
        assert isinstance(result["segments"], list) and len(result["segments"]) > 0
        assert result["num_speakers"] is None or result["num_speakers"] >= 1
        assert "spk" in result["segments"][0]

        # ---------- 4. /transcribe with formats 内嵌 ----------
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
        body2 = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="C3142.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + audio_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="formats"\r\n\r\n'
            f"srt,vtt\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        conn.request(
            "POST", "/transcribe",
            body=body2,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 200, f"formats test failed: {resp.status} {raw[:300]!r}"
        result2 = json.loads(raw)
        conn.close()
        assert "srt" in result2 and result2["srt"].startswith("1\n"), "srt format missing/malformed"
        assert "vtt" in result2 and result2["vtt"].startswith("WEBVTT"), "vtt format missing/malformed"
        print(f"[OK] /transcribe + formats: srt={len(result2['srt'])}B, vtt={len(result2['vtt'])}B")

        # ---------- 5. feature-gating: sensevoice + preset_spk_num ----------
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
        body3 = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="C3142.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + audio_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"sensevoice\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="preset_spk_num"\r\n\r\n'
            f"2\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        conn.request(
            "POST", "/transcribe",
            body=body3,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 400, f"feature-gating broken: got {resp.status}"
        assert b"spk_model" in raw, f"error msg should mention spk_model: {raw[:200]!r}"
        print(f"[OK] feature-gating: sensevoice + preset_spk_num -> 400 with proper error")
        conn.close()

        # ---------- 6. unknown model -> 404 ----------
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        body4 = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="x.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + b"\x00" * 100 + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="model"\r\n\r\n'
            f"nonexistent\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        conn.request(
            "POST", "/transcribe",
            body=body4,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 404, f"unknown model: got {resp.status}"
        print(f"[OK] unknown model -> 404")
        conn.close()

        print()
        print("=" * 70)
        print("[PASS] Phase 5.3 verification successful")
        print("=" * 70)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())