"""Adam AI Dubbing - core pipeline (CPU, free, offline after first run)."""
import os, subprocess, shutil, sys, wave
import numpy as np

# Make the bundled OpenVoice sources importable (dev layout and PyInstaller bundle).
_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, '..', 'openvoice_src'),
              os.path.join(getattr(sys, '_MEIPASS', _here), 'openvoice_src')):
    _cand = os.path.normpath(_cand)
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

# Compat: PyAV>=15 removed av.open(metadata_errors=...) which faster-whisper passes.
# Drop the kwarg so any av version works.
try:
    import av as _av
    _orig_av_open = _av.open
    def _av_open_compat(*a, **k):
        k.pop('metadata_errors', None)
        return _orig_av_open(*a, **k)
    _av.open = _av_open_compat
except ImportError:
    pass

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

# ---------- 1. transcribe ----------
_whisper_model = None
def _whisper_model_path():
    env = os.environ.get('OPENVOICE_WHISPER_MODEL')
    if env:
        return env
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'faster-whisper-base')
    local = os.path.normpath(local)
    return local if os.path.isdir(local) else 'base'

def transcribe(wav_path, log=print, initial_prompt=None):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        mp = _whisper_model_path()
        log(f'Đang tải model Whisper ({mp})...')
        _whisper_model = WhisperModel(mp, device='cpu', compute_type='int8')
    log('Đang nghe và tách lời thoại...')
    segments, _ = _whisper_model.transcribe(wav_path, language='vi', vad_filter=True,
                                             vad_parameters=dict(min_silence_duration_ms=400),
                                             initial_prompt=initial_prompt)
    out = [{'start': s.start, 'end': s.end, 'text': s.text.strip()}
           for s in segments if s.text.strip()]
    log(f'Đã tách {len(out)} câu thoại.')
    return out

# ---------- 2. vietnamese base TTS ----------
def synth_vi(text, out_mp3, voice='vi-VN-NamMinhNeural', log=print):
    import asyncio, edge_tts
    proxy = (os.environ.get('EDGE_TTS_PROXY') or os.environ.get('HTTPS_PROXY')
             or os.environ.get('https_proxy'))
    async def _go():
        await edge_tts.Communicate(text, voice, proxy=proxy).save(out_mp3)
    try:
        asyncio.run(_go())
    except Exception as e:
        raise RuntimeError(
            'Không tạo được giọng đọc (cần mạng Internet để gọi Edge-TTS). '
            f'Hãy kiểm tra mạng rồi thử lại. Chi tiết: {e}')

# ---------- 3. tone conversion to Adam ----------
_converter = None
_ref_se_cache = {}
_src_se_cache = {}
def _get_converter(ckpt_dir, log=print):
    global _converter
    if _converter is None:
        from openvoice.api import ToneColorConverter
        log('Đang tải model chuyển giọng OpenVoice (lần đầu hơi lâu)...')
        _converter = ToneColorConverter(f'{ckpt_dir}/converter/config.json', device='cpu',
                                          enable_watermark=False)
        _converter.load_ckpt(f'{ckpt_dir}/converter/checkpoint.pth')
    return _converter

def convert_to_adam(src_wav, ref_wav, out_wav, ckpt_dir, log=print, src_voice_id='base_vi'):
    from openvoice import se_extractor
    conv = _get_converter(ckpt_dir, log)
    # NOTE: vad=False -> split_audio_whisper (faster-whisper, already bundled).
    # vad=True would call whisper_timestamped.get_vad_segments(method="silero"),
    # which uses torch.hub to download snakers4/silero-vad from GitHub and asks
    # for trust via input() -> crashes in the frozen windowed app (no stdin).
    # ref_wav may be a single path or a list of paths (embeddings averaged).
    refs = list(ref_wav) if isinstance(ref_wav, (list, tuple)) else [ref_wav]
    rkey = tuple(refs)
    if rkey not in _ref_se_cache:
        import torch
        ses = [se_extractor.get_se(r, conv, vad=False)[0] for r in refs]
        _ref_se_cache[rkey] = torch.stack(ses).mean(dim=0) if len(ses) > 1 else ses[0]
        if len(ses) > 1:
            log(f'Đã gộp {len(ses)} mẫu giọng Adam làm giọng chuẩn.')
    # Base TTS voice is the same for every segment -> extract its embedding once.
    skey = (ckpt_dir, src_voice_id)
    if skey not in _src_se_cache:
        _src_se_cache[skey] = se_extractor.get_se(src_wav, conv, vad=False)[0]
    src_se = _src_se_cache[skey]
    tgt_se = _ref_se_cache[rkey]
    conv.convert(audio_src_path=src_wav, src_se=src_se, tgt_se=tgt_se,
                 output_path=out_wav, message='@MyShell')

# ---------- 4. fit to original timing ----------
def fit_duration(wav_path, target_dur, log=print):
    """Time-stretch so the clip fits the original slot. Returns possibly new path."""
    from audiotsm import wsola
    from audiotsm.io.wav import WavReader, WavWriter
    dur = wav_duration(wav_path)
    if dur <= 0 or target_dur <= 0:
        return wav_path
    ratio = target_dur / dur  # >1 means need slower/longer
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

# ---------- 5. mux ----------
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
