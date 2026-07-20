"""端到端本地功能冒烟测试。"""
import json
import os
import subprocess
import sys
import tempfile
import shutil
import urllib.request
import http.client

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PY = r"C:\Users\zhbuaa0\.conda\envs\paraformer-asr\python.exe"
ROOT = r"F:\Program Files\openclaw\workspace\paraformer-asr"
AUDIO = r"C:\Users\zhbuaa0\.openclaw\media\inbound\C3142---81c7bbb6-da1e-42ce-a3e5-f07afb900f20.wav"
RESULT_JSON = r"F:\Program Files\openclaw\workspace\paraformer-asr\C3142_result.json"

PASS, FAIL = 0, 0


def chk(name, ok, detail=""):
    global PASS, FAIL
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}", end="")
    if detail:
        print(f"  ({detail})", end="")
    print()
    if ok:
        PASS += 1
    else:
        FAIL += 1


def run_cli(*args):
    env = {**os.environ, "PYTHONPATH": ROOT, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [PY, "-m", "subforge.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def post_transcribe(fields):
    """POST /transcribe with audio + extra fields. Returns (status, json_body)."""
    boundary = "----LocalTest"
    with open(AUDIO, "rb") as f:
        audio_bytes = f.read()
    parts = [(("--" + boundary).encode())]
    parts.append(f'Content-Disposition: form-data; name="audio"; filename="C3142.wav"'.encode())
    parts.append(b"Content-Type: audio/wav")
    parts.append(b"")
    parts.append(audio_bytes)
    for k, v in fields.items():
        parts.append(("--" + boundary).encode())
        parts.append(f'Content-Disposition: form-data; name="{k}"'.encode())
        parts.append(b"")
        parts.append(str(v).encode())
    parts.append(("--" + boundary + "--").encode())
    parts.append(b"")
    body = b"\r\n".join(parts)
    conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=600)
    conn.request("POST", "/transcribe", body=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        body_json = json.loads(raw)
    except Exception:
        body_json = {}
    return resp.status, body_json, raw


# ---- TEST 1: asr models ----
print("=" * 70)
print("TEST 1: asr models (CLI)")
print("=" * 70)
r = run_cli("models")
chk("exit 0", r.returncode == 0)
chk("列出 paraformer-zh", "paraformer-zh" in r.stdout)
chk("列出 seaco-paraformer-zh", "seaco-paraformer-zh" in r.stdout)
chk("列出 sensevoice", "sensevoice" in r.stdout)
chk("列出 paraformer-zh-streaming", "paraformer-zh-streaming" in r.stdout)

# ---- TEST 2: asr polish (CLI, no key) ----
print()
print("=" * 70)
print("TEST 2: asr polish (CLI, no key)")
print("=" * 70)
r = run_cli("polish", RESULT_JSON)
chk("exit 2 (错误退出)", r.returncode == 2)
chk("明确报错 MINIMAX_API_KEY", "MINIMAX_API_KEY" in r.stderr or "MINIMAX_API_KEY" in r.stdout)

# ---- TEST 3: asr export ----
print()
print("=" * 70)
print("TEST 3: asr export (CLI)")
print("=" * 70)
tmpdir = tempfile.mkdtemp()
try:
    r = run_cli("export", RESULT_JSON, "-f", "srt,txt", "--output-dir", tmpdir)
    chk("exit 0", r.returncode == 0)
    chk("生成 .srt", os.path.exists(os.path.join(tmpdir, "C3142_result.srt")))
    chk("生成 .txt", os.path.exists(os.path.join(tmpdir, "C3142_result.txt")))
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# ---- TEST 4: /health ----
print()
print("=" * 70)
print("TEST 4: /health")
print("=" * 70)
try:
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2).read())
    chk("status ok", h.get("status") == "ok")
    chk("default_model", bool(h.get("default_model")))
    chk("ffmpeg_available 字段存在", "ffmpeg_available" in h)
except Exception as e:
    chk("reachable", False, str(e))

# ---- TEST 5: /models 含 polish 段 ----
print()
print("=" * 70)
print("TEST 5: /models 含 polish 段")
print("=" * 70)
m = json.loads(urllib.request.urlopen("http://127.0.0.1:8000/models", timeout=2).read())
chk("default 模型 = paraformer-zh", m.get("default") == "paraformer-zh")
chk("4 个模型", len(m.get("models", [])) == 4)
chk("polish 段存在", "polish" in m)
chk("polish default=minimax", m["polish"].get("default") == "minimax")
chk("polish 3 个 provider", len(m["polish"].get("providers", [])) == 3)

# ---- TEST 6: /transcribe 无 polish ----
print()
print("=" * 70)
print("TEST 6: /transcribe 无 polish")
print("=" * 70)
status, data, _ = post_transcribe({"model": "paraformer-zh", "formats": "txt"})
chk("status 200", status == 200)
chk("segments=190", len(data.get("segments", [])) == 190)
chk("num_speakers=2", data.get("num_speakers") == 2)
chk("text_len≈1631", 1620 <= len(data.get("text", "")) <= 1640)
chk("txt 内嵌存在", "txt" in data and len(data["txt"]) > 0)
chk("无 polished (未请求)", "polished" not in data)

# ---- TEST 7: /transcribe 带 polish=minimax (无 key) ----
print()
print("=" * 70)
print("TEST 7: /transcribe polish=minimax (无 key)")
print("=" * 70)
status, _, raw = post_transcribe({"model": "paraformer-zh", "formats": "srt", "polish": "minimax"})
chk("status 503", status == 503)
chk("明确报 MINIMAX_API_KEY", b"MINIMAX_API_KEY" in raw)

# ---- TEST 8: /transcribe 带 polish=unknown ----
print()
print("=" * 70)
print("TEST 8: /transcribe polish=unknown")
print("=" * 70)
status, _, raw = post_transcribe({"model": "paraformer-zh", "polish": "unknown"})
chk("status 400", status == 400)
chk("列出可用 providers", b"minimax" in raw and b"deepseek" in raw)

# ---- TEST 9: Web UI HTML ----
print()
print("=" * 70)
print("TEST 9: Web UI HTML")
print("=" * 70)
html = urllib.request.urlopen("http://127.0.0.1:8000/").read().decode("utf-8")
checks = [
    ('checkbox use-polish', 'id="use-polish"' in html),
    ('polish-hint', 'id="polish-hint"' in html),
    ('result-polished 面板', 'id="result-polished"' in html),
    ('polished-text textarea', 'id="polished-text"' in html),
    ('renderPolished 函数', 'function renderPolished' in html),
    ('usePolishCb 变量', 'usePolishCb' in html),
    ('JS 读 polishAvailable', 'polishAvailable' in html),
    ('download polished_<fmt>', 'polished_' in html),
]
for n, ok in checks:
    chk(n, ok)

print()
print("=" * 70)
print(f"TOTAL: {PASS} pass / {FAIL} fail")
print("=" * 70)
sys.exit(0 if FAIL == 0 else 1)