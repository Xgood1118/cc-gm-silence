# -*- coding: utf-8 -*-
"""静音检测。

流程：50ms 窗口、10ms 步长计算每帧 RMS → 5 帧中值平滑 → 连续低于阈值的
帧合并为候选静音段 → 丢弃短于 min_silence_ms 的段 → 段边界以 1ms 步长
修正 → 钳制在 [0, 音频时长] 内。
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


def zero_crossings(samples, start, length):
    """统计 [start, start + length) 区间内的过零次数。"""
    end = min(start + length, len(samples))
    count = 0
    for i in range(max(1, start + 1), end):
        if (samples[i - 1] < 0) != (samples[i] < 0):
            count += 1
    return count


def _median(values):
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _median_smooth(values, width=5):
    """中值平滑：去除个别帧的能量毛刺/跌落，同时保留段的边缘。"""
    half = width // 2
    return [_median(values[max(0, i - half):i + half + 1])
            for i in range(len(values))]


def _frame_classification(wav, threshold, window_ms):
    """返回 (帧起点样本列表, 窗口样本数, 每帧是否静音)。"""
    rate = wav.rate
    n = len(wav.samples)
    win = max(1, int(rate * window_ms / 1000.0))
    hop = max(1, int(rate * 0.010))  # 10ms 步长
    starts = list(range(0, max(1, n - win + 1), hop))
    rms = [frame_rms(wav.samples, s, win) for s in starts]
    smoothed = _median_smooth(rms, 5)
    return starts, win, [v < threshold for v in smoothed]


def _refine_start(samples, rate, approx_ms, threshold):
    """从粗边界向回以 1ms 步长扫描，找最后一个能量达到阈值的 10ms 探针
    窗口，精确起点即该窗口之后 1ms。"""
    probe = max(1, int(rate * 0.010))
    step = max(1, int(rate * 0.001))
    center = min(max(0, int(approx_ms * rate / 1000.0)),
                 max(0, len(samples) - probe))
    lo = max(0, center - 60 * step)
    t = center
    while t >= lo:
        if frame_rms(samples, t, probe) >= threshold:
            return (t + step) * 1000.0 / rate
        t -= step
    return lo * 1000.0 / rate


def _refine_end(samples, rate, approx_ms, threshold):
    """从粗边界向前以 1ms 步长扫描，找第一个能量达到阈值的 10ms 探针
    窗口，精确终点即该窗口之前 1ms。"""
    probe = max(1, int(rate * 0.010))
    step = max(1, int(rate * 0.001))
    center = min(max(0, int(approx_ms * rate / 1000.0)), len(samples))
    hi = min(len(samples), center + 60 * step)
    t = center
    while probe <= t <= hi:
        if frame_rms(samples, t - probe, probe) >= threshold:
            return (t - step) * 1000.0 / rate
        t += step
    return hi * 1000.0 / rate


def detect_silence(wav, threshold=200.0, window_ms=50, min_silence_ms=200):
    """检测静音段，返回 [{start_ms, end_ms, duration_ms, rms}, ...]。

    相邻窗口经中值平滑，连续低于阈值的窗口合并为一段，短于 min_silence_ms
    的段被丢弃，边界以 1ms 精度修正并钳制在音频时长内。
    """
    rate = wav.rate
    n = len(wav.samples)
    total_ms = n * 1000.0 / rate
    starts, win, silent = _frame_classification(wav, threshold, window_ms)
    # 合并连续的静音帧
    candidates = []
    i = 0
    while i < len(starts):
        if not silent[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(starts) and silent[j + 1]:
            j += 1
        start_ms = starts[i] * 1000.0 / rate
        end_ms = min((starts[j] + win) * 1000.0 / rate, total_ms)
        if end_ms - start_ms >= min_silence_ms:
            candidates.append((start_ms, end_ms))
        i = j + 1
    # 修正边界并钳制
    segments = []
    for start_ms, end_ms in candidates:
        rs = min(max(_refine_start(wav.samples, rate, start_ms, threshold),
                     0.0), total_ms)
        re = min(max(_refine_end(wav.samples, rate, end_ms, threshold),
                     0.0), total_ms)
        if re - rs < min_silence_ms:
            continue
        s0 = int(rs * rate / 1000.0)
        s1 = min(n, int(re * rate / 1000.0))
        segments.append({
            "start_ms": round(rs, 1),
            "end_ms": round(re, 1),
            "duration_ms": round(re - rs, 1),
            "rms": round(frame_rms(wav.samples, s0, max(1, s1 - s0)), 2),
        })
    return segments


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
    duration_ms = len(wav.samples) * 1000.0 / wav.rate

    silence_segs = detect_silence(wav, args.threshold, 50, args.min_ms)
    # 不做最短时长过滤再跑一遍，用其补集切出说话段，
    # 这样被过滤的短停顿仍可作为说话段的分界。
    all_silence = detect_silence(wav, args.threshold, 50, 0)
    speech_ranges = []
    prev = 0.0
    for s in all_silence:
        if s["start_ms"] - prev >= 1.0:
            speech_ranges.append((prev, s["start_ms"]))
        prev = max(prev, s["end_ms"])
    if duration_ms - prev >= 1.0:
        speech_ranges.append((prev, duration_ms))

    def enrich(seg):
        s0 = int(seg["start_ms"] * wav.rate / 1000.0)
        s1 = min(len(wav.samples), int(seg["end_ms"] * wav.rate / 1000.0))
        dur = max(1e-9, (s1 - s0) / float(wav.rate))
        seg["zero_crossing_rate"] = round(
            zero_crossings(wav.samples, s0, s1 - s0) / dur, 1)
        return seg

    speech_segs = []
    for a, b in speech_ranges:
        s0 = int(a * wav.rate / 1000.0)
        s1 = min(len(wav.samples), int(b * wav.rate / 1000.0))
        speech_segs.append(enrich({
            "start_ms": round(a, 1),
            "end_ms": round(b, 1),
            "duration_ms": round(b - a, 1),
            "rms": round(frame_rms(wav.samples, s0, max(1, s1 - s0)), 2),
        }))

    report = {
        "file": args.wav,
        "rate": wav.rate,
        "duration_ms": round(duration_ms, 1),
        "threshold": args.threshold,
        "min_silence_ms": args.min_ms,
        "silence_segments": [enrich(dict(s)) for s in silence_segs],
        "speech_segments": speech_segs,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    total = sum(s["duration_ms"] for s in silence_segs) / 1000.0
    print("检测到 %d 段静音，总时长 %.2f 秒" % (len(silence_segs), total))
    for s in report["silence_segments"]:
        print("  静音 %7.1fms - %7.1fms  时长 %6.1fms  平均能量 %.2f  过零率 %.1f/s"
              % (s["start_ms"], s["end_ms"], s["duration_ms"],
                 s["rms"], s["zero_crossing_rate"]))
    for s in speech_segs:
        print("  说话 %7.1fms - %7.1fms  时长 %6.1fms  平均能量 %.2f  过零率 %.1f/s"
              % (s["start_ms"], s["end_ms"], s["duration_ms"],
                 s["rms"], s["zero_crossing_rate"]))
    return 0
