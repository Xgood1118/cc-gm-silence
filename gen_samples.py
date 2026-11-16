# -*- coding: utf-8 -*-
"""生成一段音：说话段 0.4s、静音 0.6s、说话 0.3s、极短静音 0.08s、说话 0.5s。"""
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
RATE = 16000


def tone(n, freq=300, amp=9000):
    return [int(amp * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n)]


def build():
    out = []
    out += tone(int(RATE * 0.4))
    out += [0] * int(RATE * 0.6)
    out += tone(int(RATE * 0.3), 420)
    out += [0] * int(RATE * 0.08)
    out += tone(int(RATE * 0.5), 260)
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    samples = build()
    data = struct.pack("<%dh" % len(samples), *samples)
    fmt = struct.pack("<HHIIHH", 1, 1, RATE, RATE * 2, 2, 16)
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    path = os.path.join(HERE, "data", "speech.wav")
    with open(path, "wb") as f:
        f.write(header + b"fmt " + struct.pack("<I", 16) + fmt +
                b"data" + struct.pack("<I", len(data)) + data)
    print("已生成", path, "%.2f 秒" % (len(samples) / RATE))
