"""legacy make_md.py —— 兼容旧入口。

行为等价于原 make_md.py：从 C3142_result.json 生成 C3142_transcript.md。
"""
import sys

from subforge.cli import main

OUT_JSON = r"F:\Program Files\openclaw\workspace\paraformer-asr\C3142_result.json"
OUT_MD = r"F:\Program Files\openclaw\workspace\paraformer-asr\C3142_transcript.md"


if __name__ == "__main__":
    sys.exit(main([
        "export", OUT_JSON,
        "--format", "md",
        "--output", OUT_MD,
    ]))