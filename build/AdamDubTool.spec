# -*- mode: python ; coding: utf-8 -*-
import os

BUILD = r'Z:\home\hatch\workspace\adam-dub\build'
SRC = r'Z:\home\hatch\workspace\adam-dub\src'

datas = [
    (os.path.join(BUILD, 'openvoice_pkg', 'openvoice'), os.path.join('openvoice_src', 'openvoice')),
    (os.path.join(BUILD, 'models', 'checkpoints_v1'), os.path.join('models', 'checkpoints_v1')),
    (os.path.join(BUILD, 'models', 'faster-whisper-base'), os.path.join('models', 'faster-whisper-base')),
    (os.path.join(BUILD, 'ref'), 'ref'),
    (os.path.join(BUILD, 'torch_hub'), 'torch_hub'),
]
binaries = [
    (os.path.join(BUILD, 'ffmpeg.exe'), '.'),
]

hiddenimports = [
    'openvoice', 'openvoice.api', 'openvoice.se_extractor', 'openvoice.mel_processing',
    'openvoice.models', 'openvoice.modules', 'openvoice.transforms', 'openvoice.attentions',
    'openvoice.commons', 'openvoice.utils', 'openvoice.text', 'openvoice.text.cleaners',
    'openvoice.text.symbols', 'openvoice.text.english', 'openvoice.text.mandarin',
    'edge_tts', 'faster_whisper', 'ctranslate2', 'onnxruntime', 'tokenizers',
    'audiotsm', 'audiotsm.io.wav', 'librosa', 'soundfile', 'scipy', 'sklearn',
    'whisper_timestamped', 'pydub', 'jieba', 'pypinyin', 'cn2an', 'eng_to_ipa',
    'unidecode', 'inflect', 'av', 'aiohttp', 'certifi',
]

a = Analysis(
    [os.path.join(SRC, 'app.py')],
    pathex=[SRC, os.path.join(BUILD, 'openvoice_pkg')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(BUILD, 'hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['gradio', 'matplotlib', 'IPython', 'notebook'],
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
