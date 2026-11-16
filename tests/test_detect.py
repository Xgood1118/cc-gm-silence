# -*- coding: utf-8 -*-
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silence.core import Wav, detect_silence, zero_crossings  # noqa: E402

RATE = 16000


def tone(n, freq=300, amp=9000):
    return [int(amp * math.sin(2 * math.pi * freq * i / RATE)) for i in range(n)]


def build_speech():
    """与 data/speech.wav 同结构：0.4s 说话、0.6s 静音、0.3s 说话、
    0.08s 停顿、0.5s 说话。"""
    out = []
    out += tone(int(RATE * 0.4))
    out += [0] * int(RATE * 0.6)
    out += tone(int(RATE * 0.3), 420)
    out += [0] * int(RATE * 0.08)
    out += tone(int(RATE * 0.5), 260)
    return Wav(RATE, out)


def test_long_silence_merged_into_one_segment():
    """0.6 秒的静音必须合并成一段，而不是十几个窗口条目。"""
    segs = detect_silence(build_speech(), 200.0, 50, 200)
    assert len(segs) == 1
    assert abs(segs[0]["duration_ms"] - 600) <= 20


def test_short_pause_filtered():
    """0.08 秒的停顿短于 min_silence_ms，不能算一段静音。"""
    segs = detect_silence(build_speech(), 200.0, 50, 200)
    for s in segs:
        assert not (1200 <= s["start_ms"] <= 1400)
        assert s["duration_ms"] >= 200


def test_boundary_error_within_10ms():
    """段起止与真实边界（400ms / 1000ms）的误差不超过 10 毫秒。"""
    segs = detect_silence(build_speech(), 200.0, 50, 200)
    assert len(segs) == 1
    assert abs(segs[0]["start_ms"] - 400) <= 10
    assert abs(segs[0]["end_ms"] - 1000) <= 10


def test_speech_not_misdetected():
    """正常音量的说话段（含能量起伏）不能被误判成静音。"""
    segs = detect_silence(build_speech(), 200.0, 50, 200)
    for s in segs:
        # 说话区间 0-400、1000-1300、1380-1880 内不能有静音段
        for a, b in [(0, 400), (1000, 1300), (1380, 1880)]:
            assert s["end_ms"] <= a or s["start_ms"] >= b


def test_threshold_configurable():
    """阈值可配置：阈值为 0 时任何段都不低于阈值。"""
    assert detect_silence(build_speech(), 0.0, 50, 200) == []


def test_zero_crossings_respects_interval():
    """过零率只统计传入的 [start, start+length) 区间。"""
    s = [1, -1, 1, -1, 5, 5, 5, -5]
    assert zero_crossings(s, 0, 4) == 3   # 前 4 个样本 3 次过零
    assert zero_crossings(s, 4, 4) == 1   # 后 4 个样本 1 次过零
    assert zero_crossings(s, 4, 100) == 1  # 超出长度被截断，不重数


def test_zero_crossings_unvoiced_vs_voiced():
    """清音（类噪声）的过零率应远高于浊音（周期信号）。"""
    n = int(RATE * 0.2)
    voiced = tone(n, 300)
    unvoiced = [9000 if i % 2 else -9000 for i in range(n)]
    z_voiced = zero_crossings(voiced, 0, n)
    z_unvoiced = zero_crossings(unvoiced, 0, n)
    assert z_unvoiced > 10 * z_voiced


def test_boundaries_clamped_to_duration():
    """静音在音频开头/结尾时，边界不能超出 [0, 音频时长]。"""
    samples = ([0] * int(RATE * 0.5) + tone(int(RATE * 0.3))
               + [0] * int(RATE * 0.4))
    wav = Wav(RATE, samples)
    total_ms = len(samples) * 1000.0 / RATE
    segs = detect_silence(wav, 200.0, 50, 200)
    assert len(segs) == 2
    assert 0 <= segs[0]["start_ms"] <= 10
    assert abs(segs[-1]["end_ms"] - total_ms) <= 10
    for s in segs:
        assert 0 <= s["start_ms"] < s["end_ms"] <= total_ms
