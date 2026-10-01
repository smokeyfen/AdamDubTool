"""Adam AI Dubbing v7 - core pipeline (CPU, free, offline).

Whisper large-v3-turbo -> auto emotion tagging -> VieNeu-TTS v3 Turbo
(direct Adam voice cloning) -> warm voice post-processing -> FFmpeg.
No manual transcript review: emotion tags are added automatically.
"""
import os, subprocess, sys, wave
import numpy as np

# ---------- audio helpers ----------
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def wav_duration(path):
    with wave.open(path, 'rb') as w:
        return w.getnframes() / w.getframerate()

def ensure_wav_16k_mono(src, dst):
    run(['ffmpeg', '-y', '-i', src, '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', dst])

def extract_audio(video_path, out_wav):
    ensure_wav_16k_mono(video_path, out_wav)

# ---------- 1. transcribe (faster-whisper large-v3-turbo) ----------
_whisper_model = None
def _models_dir(*parts):
    """Locate the bundled models dir in frozen (ROOT/models) or dev (ROOT/build/models) layout."""
    here = os.path.dirname(os.path.abspath(__file__))
    frozen_root = getattr(sys, '_MEIPASS', None)
    dev_root = os.path.normpath(os.path.join(here, '..'))
    for root in ([frozen_root] if frozen_root else []) + [dev_root]:
        for base in ('models', os.path.join('build', 'models')):
            cand = os.path.join(root, base, *parts)
            if os.path.isdir(cand):
                return cand
    # Fallback: frozen layout path (the app sets env vars to this).
    return os.path.join(frozen_root or dev_root, 'models', *parts)

def _whisper_model_select():
    """Return (dir, label). Prefer large-v3-turbo if fully present, else base.

    large-v3-turbo (~1.5GB) is the accurate model; base is the offline fallback.
    A directory only counts if model.bin actually exists (partial downloads
    must never be used).
    """
    for name, label in (('faster-whisper-large-v3-turbo', 'large-v3-turbo'),
                        ('faster-whisper-base', 'base')):
        d = _models_dir(name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'model.bin')):
            return d, label
    return None, None

def transcribe(wav_path, log=print, initial_prompt=None):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        mp = os.environ.get('ADAM_WHISPER_MODEL')
        label = 'large-v3-turbo'
        if not (mp and os.path.exists(os.path.join(mp, 'model.bin'))):
            mp, label = _whisper_model_select()
        if not mp:
            mp, label = 'large-v3-turbo', 'large-v3-turbo'  # last resort: download on first run
        log(f'Đang tải model Whisper {label} ({mp})...')
        _whisper_model = WhisperModel(mp, device='cpu', compute_type='int8')
    log('Đang nghe và tách lời thoại (large-v3-turbo)...')
    segments, _ = _whisper_model.transcribe(wav_path, language='vi', vad_filter=True,
                                             vad_parameters=dict(min_silence_duration_ms=400),
                                             initial_prompt=initial_prompt)
    out = [{'start': s.start, 'end': s.end, 'text': s.text.strip()}
           for s in segments if s.text.strip()]
    log(f'Đã tách {len(out)} câu thoại.')
    return out

# ---------- 2. Adam voice via VieNeu-TTS v3 Turbo (direct cloning) ----------
_tts = None
def _vieneu_root():
    vroot = _models_dir('vieneu')
    if not os.path.isdir(vroot):
        raise RuntimeError('Không tìm thấy thư mục models/vieneu trong gói cài đặt.')
    return vroot

def get_tts(log=print):
    """Lazy singleton. Offline: all model files are bundled locally."""
    global _tts
    if _tts is None:
        from vieneu import Vieneu
        from vieneu._v3_turbo_engine import onnx_runtime_lite as _O
        vroot = _vieneu_root()
        # The codec artifacts live in the bundled dir; keep HF out of the loop.
        _orig_fetch = _O.OnnxV3LiteEngine._fetch
        def _local_fetch(repo, files, subfolder=None):
            if 'MOSS' in str(repo):
                # Must be a pathlib.Path: the engine does `cd / "file.onnx"`.
                from pathlib import Path as _P
                return _P(os.path.join(vroot, 'codec'))
            return _orig_fetch(repo, files, subfolder)
        _O.OnnxV3LiteEngine._fetch = staticmethod(_local_fetch)
        try:
            log('Đang tải model giọng nói VieNeu v3 Turbo (lần đầu hơi lâu)...')
            _tts = Vieneu(backend='onnx', checkpoint_path=vroot,
                          onnx_dir=os.path.join(vroot, 'onnx_update'))
        finally:
            _O.OnnxV3LiteEngine._fetch = _orig_fetch
    return _tts

def enroll_adam(ref_wav, log=print):
    """Register the Adam reference voice once per run. Returns the voice name."""
    tts = get_tts(log)
    if 'Adam' not in tts._preset_voices:
        log('Đang đăng ký giọng mẫu Adam (lấy 8s đầu, khử nhiễu)...')
        tts.add_voice('Adam', ref_wav, denoise=True)
        log('Đã đăng ký giọng Adam.')
    return 'Adam'

def synth_adam(text, out_wav_16k, log=print):
    """Direct text -> Adam voice (48kHz) -> 16k mono wav for the dub track."""
    tts = get_tts(log)
    wav48 = tts.infer(text, voice='Adam', apply_watermark=False)
    if wav48 is None or len(wav48) == 0:
        raise RuntimeError(f'Không tạo được audio cho câu: "{text[:60]}"')
    tmp48 = out_wav_16k + '.48k.wav'
    tts.save(wav48, tmp48)
    try:
        ensure_wav_16k_mono(tmp48, out_wav_16k)
    finally:
        try: os.remove(tmp48)
        except OSError: pass

# ---------- 3. fit to original timing ----------
def fit_duration(wav_path, target_dur, log=print):
    """Time-stretch so the clip fits the original slot. Returns possibly new path."""
    from audiotsm import wsola
    from audiotsm.io.wav import WavReader, WavWriter
    dur = wav_duration(wav_path)
    if dur <= 0 or target_dur <= 0:
        return wav_path
    speed = dur / target_dur  # wsola speed: >1 = faster
    if 0.85 <= speed <= 1.18:
        out = wav_path.replace('.wav', '_fit.wav')
        with WavReader(wav_path) as r, WavWriter(out, r.channels, r.samplerate) as w:
            wsola(r.channels, speed=speed).run(r, w)
        return out
    log(f'  (câu dài/ngắn bất thường, giữ nguyên tốc độ: {dur:.1f}s -> slot {target_dur:.1f}s)')
    return wav_path

def build_track(clips, total_dur, out_wav, sr=16000, log=print):
    """clips: list of (start_sec, wav_path). Place each at its start, pad silence."""
    track = np.zeros(int(total_dur * sr) + sr, dtype=np.float32)
    for start, path in clips:
        with wave.open(path, 'rb') as w:
            assert w.getframerate() == sr and w.getnchannels() == 1
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        i0 = int(start * sr)
        i1 = min(i0 + len(data), len(track))
        track[i0:i1] += data[:i1 - i0]
    track = np.clip(track, -1, 1)
    with wave.open(out_wav, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((track * 32767).astype(np.int16).tobytes())
    log(f'Đã dựng track lồng tiếng: {wav_duration(out_wav):.1f}s')

# ---------- 4. mux ----------
def _has_audio_stream(video_path):
    try:
        out = subprocess.run(['ffmpeg', '-hide_banner', '-i', video_path],
                             capture_output=True, text=True).stderr
        return 'Audio:' in out
    except Exception:
        return True  # assume yes; ffmpeg will report the real error

def mux(video_path, dub_wav, out_mp4, orig_volume=0.12, log=print):
    tmp = out_mp4 + '.tmp.m4a'
    run(['ffmpeg', '-y', '-i', dub_wav, '-c:a', 'aac', '-b:a', '160k', tmp])
    if _has_audio_stream(video_path) and orig_volume > 0:
        run(['ffmpeg', '-y', '-i', video_path, '-i', tmp,
             '-filter_complex', f'[0:a]volume={orig_volume}[a0];[1:a][a0]amix=inputs=2:duration=first[a]',
             '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
             '-shortest', out_mp4])
    else:
        if not _has_audio_stream(video_path):
            log('Video gốc không có tiếng; chỉ dùng track lồng tiếng.')
        run(['ffmpeg', '-y', '-i', video_path, '-i', tmp,
             '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
             '-shortest', out_mp4])
    os.remove(tmp)
    log(f'Xong! File đầu ra: {out_mp4}')
