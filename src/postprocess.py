"""v7 - Post-processing cho giọng lồng tiếng: ấm hơn, rõ hơn, "người" hơn.

Chỉ dùng numpy (đã có sẵn trong bundle) - không thêm dependency mới.
Chuỗi xử lý trên track 16kHz mono:
  1. High-pass 80Hz  : cắt ù, rè tần thấp
  2. Peaking -2dB @300Hz : bớt "ồm", giọng thoáng hơn
  3. Peaking +3dB @4kHz  : tăng độ rõ, giọng "sáng" như thu phòng thu
  4. Compressor mềm (3:1, threshold -20dB): giọng đều, nhấn nhá rõ ràng
  5. Makeup gain + limiter -1dBFS: to đều, không vỡ tiếng
"""

import wave
import numpy as np


def _biquad(x, b0, b1, b2, a0, a1, a2):
    """Lọc IIR bậc 2 dạng Direct Form I (numpy thuần)."""
    b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
    a1, a2 = a1 / a0, a2 / a0
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(len(x)):
        x0 = x[n]
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = y0
        x2, x1, y2, y1 = x1, x0, y1, y0
    return y


def highpass(x, sr, freq):
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / 2 * np.sqrt(2)  # Q ~ 0.707
    cw = np.cos(w0)
    return _biquad(x, (1 + cw) / 2, -(1 + cw), (1 + cw) / 2,
                   1 + alpha, -2 * cw, 1 - alpha)


def peaking(x, sr, freq, gain_db, Q=1.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * Q)
    cw = np.cos(w0)
    return _biquad(x, 1 + alpha * A, -2 * cw, 1 - alpha * A,
                   1 + alpha / A, -2 * cw, 1 - alpha / A)


def _compress(x, sr, thresh_db=-20.0, ratio=3.0, attack_ms=5.0, release_ms=80.0):
    """Compressor một chiều đơn giản: làm đều âm lượng, nhấn nhá rõ hơn."""
    eps = 1e-9
    env = np.zeros_like(x)
    a_att = np.exp(-1.0 / (attack_ms * sr / 1000))
    a_rel = np.exp(-1.0 / (release_ms * sr / 1000))
    peak = 0.0
    for n in range(len(x)):
        v = abs(x[n])
        a = a_att if v > peak else a_rel
        peak = a * peak + (1 - a) * v
        env[n] = peak
    env_db = 20 * np.log10(env + eps)
    over = np.clip(env_db - thresh_db, 0, None)
    gain_db = over * (1 - 1 / ratio)
    return x * 10 ** (-gain_db / 20)


def _read_wav(path):
    with wave.open(path, 'rb') as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        return data.astype(np.float32) / 32768.0


def _write_wav(path, x):
    x = np.clip(x, -1, 1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def warm_voice(wav_path, log=print):
    """Áp EQ + compression + normalize cho track lồng tiếng. Ghi đè file."""
    x = _read_wav(wav_path)
    if len(x) == 0:
        return wav_path
    sr = 16000
    x = highpass(x, sr, 80)
    x = peaking(x, sr, 300, -2.0, Q=1.0)     # bớt ồm
    x = peaking(x, sr, 4000, +3.0, Q=1.0)    # tăng độ rõ / sáng
    x = _compress(x, sr, thresh_db=-20.0, ratio=3.0)
    # Normalize về -20dBFS RMS rồi limiter -1dBFS
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    x = x * (10 ** (-20 / 20) / rms)
    peak = np.max(np.abs(x)) + 1e-9
    if peak > 10 ** (-1 / 20):
        x = x * (10 ** (-1 / 20) / peak)
    _write_wav(wav_path, x)
    log(f"Đã làm ấm + đều giọng (EQ/compressor): {wav_path}")
    return wav_path
