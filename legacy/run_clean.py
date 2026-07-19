"""legacy run_clean.py —— 兼容旧入口。

行为等价于原 run_clean.py：识别 C3142.wav，把结果写到仓库根目录的
C3142_result.json + C3142_transcript.txt。
"""
import sys

from subforge.cli import main

AUDIO = r"C:\Users\zhbuaa0\.openclaw\media\inbound\C3142---81c7bbb6-da1e-42ce-a3e5-f07afb900f20.wav"


if __name__ == "__main__":
    sys.exit(main([
        "transcribe", AUDIO,
        "--format", "json,txt,md",
        "--output-dir", ".",
    ]))