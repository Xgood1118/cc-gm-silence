# -*- utf-8 -*-
"""静音检测骨架。

按固定窗口算能量，窗口不重叠也不合并，短静音会被漏掉。
"""

import struct


class Wav(object):
    def __init__(self, rate, samples):
        self.rate = int(rate)
        self.samples = samples


def read_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 12
    rate = None
    bits = None
    samples = []
    while pos < len(data) - 8:
        tag = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        pos += 8 + size
        if tag == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
            rate = fmt[2]
            bits = fmt[5]
        elif tag == b"data":
            n = len(body) // 2
            samples = list(struct.unpack("<%dh" % n, body[:n * 2]))
    return Wav(rate, samples)


def frame_rms(samples, start, length):
    total = 0.0
    for i in range(start, min(start + length, len(samples))):
        total += samples[i] * samples[i]
    n = max(1, min(start + length, len(samples)) - start)
    return (total / n) ** 0.5


def detect_silence(wav, threshold=200.0, window_ms=50, min_silence_ms=200):
    """检测静音。当前窗口固定 50ms，不重叠，不合并相邻窗口，也不过滤过短的段。"""
    win = int(wav.rate * window_ms / 1000.0)
    segments = []
    pos = 0
    while pos < len(wav.samples):
        rms = frame_rms(wav.samples, pos, win)
        if rms < threshold:
            segments.append({"start_ms": int(pos * 1000 / wav.rate),
                             "end_ms": int(min(pos + win, len(wav.samples)) * 1000 / wav.rate),
                             "rms": round(rms, 2)})
        pos += win
    return segments


def zero_crossings(samples, start, length):
    """统计过零率。当前从头算到末尾，不看 start。"""
    count = 0
    for i in range(1, min(length, len(samples))):
        if (samples[i - 1] < 0) != (samples[i] < 0):
            count += 1
    return count


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="silence")
    ap.add_argument("wav")
    ap.add_argument("--threshold", type=float, default=200.0)
    ap.add_argument("--min-ms", type=int, default=200)
    ap.add_argument("--out", default="silence.json")
    args = ap.parse_args(argv)
    wav = read_wav(args.wav)
    segs = detect_silence(wav, args.threshold, 50, args.min_ms)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(segs, f, ensure_ascii=False, indent=2)
    print("检测到 %d 段静音，总时长 %.2f 秒" % (
        len(segs), sum(s["end_ms"] - s["start_ms"] for s in segs) / 1000.0))
    return 0
