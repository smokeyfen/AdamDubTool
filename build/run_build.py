"""Run PyInstaller under Wine (isolated child patched to avoid broken
STARTUPINFO handle_list; see patch in site-packages/PyInstaller/isolated/_parent.py)."""
from PyInstaller.__main__ import run

run([
    'AdamDubTool.spec',
    '--noconfirm',
    '--log-level', 'WARN',
    '--distpath', 'dist',
    '--workpath', 'pybuild',
])
