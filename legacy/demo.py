"""legacy demo.py —— 兼容旧入口。

行为等价于原 demo.py：
- 如果 AUDIO_PATHS 非空用本地文件列表，否则用 AUDIO_URL（默认 ModelScope 示例）
- 加载 Paraformer-large + VAD + PUNC + SPK
- 把分段结果打到 stdout
"""
import sys

from subforge.cli import main

AUDIO_PATHS = [
    # 多个文件支持一起传，自动做说话人分离
    # "path/to/your/audio1.wav",
    # "path/to/your/audio2.wav",
]
AUDIO_URL = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav"


if __name__ == "__main__":
    if AUDIO_PATHS:
        argv = ["transcribe", *AUDIO_PATHS]
    else:
        argv = ["transcribe", AUDIO_URL]
    sys.exit(main(argv))