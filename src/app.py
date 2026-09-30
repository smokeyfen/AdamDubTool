"""Adam AI Dubbing Tool - GUI (Tkinter). Free; needs internet for TTS voice generation."""
# --- MUST be first: frozen apps have no .py sources, so libraries that call
# inspect.getsource() (e.g. torch's config module at import) would crash with
# OSError('could not get source code'). Neutralize it before any other import.
import inspect as _inspect
def _safe_getsource(obj, _orig=_inspect.getsource):
    try:
        return _orig(obj)
    except (OSError, TypeError):
        return ''
def _safe_getsourcelines(obj, _orig=_inspect.getsourcelines):
    try:
        return _orig(obj)
    except (OSError, TypeError):
        return ([], 0)
_inspect.getsource = _safe_getsource
_inspect.getsourcelines = _safe_getsourcelines
del _inspect, _safe_getsource, _safe_getsourcelines

import os, sys, threading, traceback, tempfile

# --- frozen (PyInstaller) vs dev paths ---
if getattr(sys, 'frozen', False):
    ROOT = sys._MEIPASS                      # bundled data root
    _exe_dir = os.path.dirname(sys.executable)
    os.environ['PATH'] = _exe_dir + os.pathsep + ROOT + os.pathsep + os.environ.get('PATH', '')
    WORKROOT = os.path.join(tempfile.gettempdir(), 'adam_dub_work')
    FROZEN = True
else:
    ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    WORKROOT = os.path.join(ROOT, 'work')
    FROZEN = False

# Model locations must be set before pipeline/openvoice imports.
os.environ.setdefault('OPENVOICE_WHISPER_MODEL', os.path.join(ROOT, 'models', 'faster-whisper-base'))
os.environ.setdefault('TORCH_HOME', os.path.join(ROOT, 'torch_hub'))
# Sanitize broken proxy-bypass entries some sandboxes inject (breaks httpx).
for _k in ('NO_PROXY', 'no_proxy'):
    _v = os.environ.get(_k, '')
    if '[' in _v:
        os.environ[_k] = ','.join(p for p in _v.replace('[', '').replace(']', '').split(',') if p)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if not FROZEN:
    sys.path.insert(0, os.path.join(ROOT, 'src') if os.path.isdir(os.path.join(ROOT, 'src'))
                    else os.path.dirname(os.path.abspath(__file__)))
from pipeline import (extract_audio, transcribe, synth_vi, ensure_wav_16k_mono,
                      convert_to_adam, fit_duration, build_track, mux, wav_duration)

REFS = {'Adam 1': 'ref/adam1.wav', 'Adam 2': 'ref/adam2.wav', 'Adam 3': 'ref/adam3.wav'}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Adam AI Dubbing - Lồng tiếng AI giọng Adam')
        self.geometry('620x560')
        self.video = tk.StringVar()
        self.ref = tk.StringVar(value='Adam 1')
        self.keep_orig = tk.DoubleVar(value=0.12)
        self._build()

    def _build(self):
        f = ttk.Frame(self, padding=16); f.pack(fill='both', expand=True)
        ttk.Label(f, text='Video cần lồng tiếng:').pack(anchor='w')
        r1 = ttk.Frame(f); r1.pack(fill='x', pady=4)
        ttk.Entry(r1, textvariable=self.video).pack(side='left', fill='x', expand=True)
        ttk.Button(r1, text='Chọn...', command=self.pick_video).pack(side='left', padx=6)
        ttk.Label(f, text='Giọng mẫu Adam:').pack(anchor='w', pady=(8, 0))
        ttk.Combobox(f, textvariable=self.ref, values=list(REFS), state='readonly').pack(anchor='w', pady=4)
        ttk.Label(f, text='Âm lượng tiếng gốc giữ lại (0 = tắt hẳn):').pack(anchor='w', pady=(8, 0))
        ttk.Scale(f, from_=0, to=0.5, variable=self.keep_orig).pack(fill='x', pady=4)
        ttk.Button(f, text='🎙️ BẮT ĐẦU LỒNG TIẾNG', command=self.start).pack(pady=12, fill='x')
        self.prog = ttk.Progressbar(f, mode='indeterminate'); self.prog.pack(fill='x')
        self.logw = tk.Text(f, height=16, state='disabled'); self.logw.pack(fill='both', expand=True, pady=8)
        ttk.Label(f, text='Miễn phí • Cần mạng khi chạy (để tạo giọng đọc) • Model AI đã đóng gói sẵn',
                  foreground='gray').pack()

    def log(self, msg):
        # Thread-safe: worker thread schedules GUI updates on the main thread.
        self.after(0, self._log_main, msg)

    def _log_main(self, msg):
        self.logw.configure(state='normal'); self.logw.insert('end', msg + '\n')
        self.logw.see('end'); self.logw.configure(state='disabled'); self.update_idletasks()

    def pick_video(self):
        p = filedialog.askopenfilename(filetypes=[('Video', '*.mp4 *.mkv *.avi *.mov *.webm')])
        if p: self.video.set(p)

    def start(self):
        if not self.video.get():
            messagebox.showwarning('Thiếu file', 'Hãy chọn video trước.'); return
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            self.after(0, self.prog.start)
            v = self.video.get()
            keep = self.keep_orig.get()
            ref = os.path.join(ROOT, REFS[self.ref.get()])
            ckpt = os.path.join(ROOT, 'models', 'checkpoints_v1')
            work = WORKROOT; os.makedirs(work, exist_ok=True)
            name = os.path.splitext(os.path.basename(v))[0]
            out = os.path.join(os.path.dirname(v), f'{name}_adam_dubbed.mp4')

            self.log('Bước 1/5: Tách audio...')
            wav = os.path.join(work, 'orig.wav'); extract_audio(v, wav)
            total = wav_duration(wav)

            self.log('Bước 2/5: Nghe và tách lời thoại (Whisper)...')
            segs = transcribe(wav, self.log)
            if not segs:
                self.log('Không nghe được lời thoại nào. Dừng.'); return

            self.log('Bước 3/5: Tạo giọng đọc tiếng Việt...')
            clips = []
            for i, s in enumerate(segs):
                self.log(f'  Câu {i+1}/{len(segs)}: "{s["text"][:50]}..."')
                base_mp3 = os.path.join(work, f'seg{i}_base.mp3')
                base_wav = os.path.join(work, f'seg{i}_base.wav')
                synth_vi(s['text'], base_mp3, log=self.log)
                ensure_wav_16k_mono(base_mp3, base_wav)
                self.log('  -> Đổi sang giọng Adam...')
                adam_wav = os.path.join(work, f'seg{i}_adam.wav')
                convert_to_adam(base_wav, ref, adam_wav, ckpt, self.log)
                fitted = fit_duration(adam_wav, s['end'] - s['start'] + 0.15, self.log)
                final = os.path.join(work, f'seg{i}_final.wav')
                ensure_wav_16k_mono(fitted, final)
                clips.append((s['start'], final))

            self.log('Bước 4/5: Dựng track lồng tiếng khớp nhịp...')
            dub = os.path.join(work, 'dubbed.wav')
            build_track(clips, total, dub, log=self.log)

            self.log('Bước 5/5: Ghép vào video...')
            mux(v, dub, out, orig_volume=keep, log=self.log)
            self.after(0, messagebox.showinfo, 'Xong', f'Hoàn tất!\n{out}')
        except Exception as e:
            tb = traceback.format_exc()
            try:
                logf = os.path.join(_exe_dir if FROZEN else WORKROOT, 'adam_dub_error.log')
                with open(logf, 'a', encoding='utf-8') as f:
                    f.write(tb + '\n')
                errmsg = str(e) + '\n\n(chi tiết lỗi đã ghi vào adam_dub_error.log)'
            except Exception:
                errmsg = str(e)
            self.after(0, messagebox.showerror, 'Lỗi', errmsg)
        finally:
            self.after(0, self.prog.stop)

if __name__ == '__main__':
    App().mainloop()
