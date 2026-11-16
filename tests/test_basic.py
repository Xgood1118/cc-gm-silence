# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silence.core import Wav, frame_rms, zero_crossings  # noqa: E402


def test_frame_rms_zero():
    assert frame_rms([0] * 100, 0, 50) == 0


def test_frame_rms_value():
    s = [100] * 40
    assert abs(frame_rms(s, 0, 40) - 100) < 1e-6


def test_zero_crossings():
    s = [1, -1, 1, -1]
    assert zero_crossings(s, 0, 4) == 3


def test_wav_container():
    w = Wav(16000, [0] * 100)
    assert w.rate == 16000
