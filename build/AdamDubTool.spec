# -*- mode: python ; coding: utf-8 -*-
# AdamDubTool v7: Whisper large-v3-turbo + auto emotion tagging + VieNeu-TTS v3 Turbo
# (CPU/ONNX, torch-free) + warm voice post-processing.
import os

BUILD = r'Z:\home\hatch\workspace\adam-dub\build'
SRC = r'Z:\home\hatch\workspace\adam-dub\src'
SITE = r'Z:\home\hatch\.wine\drive_c\Python310\Lib\site-packages'

def _d(src, dst):
    return (src, dst) if os.path.isdir(src) else None

_whisper_datas = [
    _d(os.path.join(BUILD, 'models', 'faster-whisper-base'),
       os.path.join('models', 'faster-whisper-base')),
    # Only bundled when fully downloaded (model.bin present).
    _d(os.path.join(BUILD, 'models', 'faster-whisper-large-v3-turbo'),
       os.path.join('models', 'faster-whisper-large-v3-turbo'))
    if os.path.exists(os.path.join(BUILD, 'models', 'faster-whisper-large-v3-turbo', 'model.bin'))
    else None,
]

datas = [
    # Whisper STT model: base is bundled (offline). If a complete
    # faster-whisper-large-v3-turbo dir is present it is preferred instead.
    *[_x for _x in _whisper_datas if _x],
    (os.path.join(BUILD, 'models', 'vieneu'), os.path.join('models', 'vieneu')),
    (os.path.join(BUILD, 'ref'), 'ref'),
    # faster-whisper's bundled Silero VAD model (PyInstaller misses it).
    (os.path.join(SITE, 'faster_whisper', 'assets', 'silero_vad_v6.onnx'),
     os.path.join('faster_whisper', 'assets')),
    # VieNeu preset-voice registry (loaded from package dir at runtime).
    (os.path.join(SITE, 'vieneu', 'assets', 'voices_v3_turbo.json'),
     os.path.join('vieneu', 'assets')),
]
binaries = [
    (os.path.join(BUILD, 'ffmpeg.exe'), '.'),
]

hiddenimports = [
    'faster_whisper', 'ctranslate2', 'onnxruntime', 'tokenizers', 'av',
    'soundfile', 'audiotsm', 'audiotsm.io.wav',
    'vieneu', 'vieneu.factory', 'vieneu.v3turbo', 'vieneu.base',
    'vieneu._v3_turbo_engine.onnx_runtime_lite',
    'vieneu._v3_turbo_engine.onnx_denoiser',
    'vieneu._v3_turbo_engine.speaker.onnx_extractor',
    'vieneu_utils', 'vieneu_utils.phonemize_text', 'vieneu_utils.core_utils',
    'huggingface_hub', 'librosa', 'numba', 'scipy', 'numpy',
    'kaldi_native_fbank', 'sea_g2p', 'soxr', 'yaml', 'tqdm', 'packaging',
]

a = Analysis(
    [os.path.join(SRC, 'app.py')],
    pathex=[SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['gradio', 'matplotlib', 'IPython', 'notebook', 'torch', 'torchaudio',
              'torchgen', 'transformers', 'whisper_timestamped', 'openvoice',
              'edge_tts', 'jieba', 'pypinyin', 'cn2an', 'eng_to_ipa', 'trafilatura'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AdamDubTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AdamDubTool',
)
