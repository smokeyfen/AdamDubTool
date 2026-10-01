"""Adam AI Dubbing Tool v7 - GUI (Tkinter). Free; fully offline.

Whisper large-v3-turbo -> auto emotion tagging -> VieNeu-TTS v3 Turbo
(direct Adam voice cloning) -> warm voice post-processing -> FFmpeg.
No manual transcript review step.
"""
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

# Model locations: pipeline._whisper_model_select() picks large-v3-turbo when a
# complete copy is bundled, otherwise falls back to the bundled base model.
# (ADAM_WHISPER_MODEL env var can override the choice with an explicit dir.)
# Force Hugging Face libraries to stay offline: every model file is bundled.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
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
from pipeline import (extract_audio, transcribe, ensure_wav_16k_mono,
                      enroll_adam, synth_adam, fit_duration, build_track,
                      mux, wav_duration)
from emotion import auto_tag_segments
from postprocess import warm_voice

REFS = {'Tự động: Adam 2 (test tốt nhất)': 'ref/adam2.wav',
        'Adam 1': 'ref/adam1.wav',
        'Adam 2': 'ref/adam2.wav',
        'Adam 3': 'ref/adam3.wav'}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Adam AI Dubbing v7 - Lồng tiếng AI giọng Adam')
        self.geometry('620x600')
        self.video = tk.StringVar()
        self.ref = tk.StringVar(value='Tự động: Adam 2 (test tốt nhất)')
        self.keywords = tk.StringVar()
        self.keep_orig = tk.DoubleVar(value=0.12)
        self.auto_emotion = tk.BooleanVar(value=True)
        self._build()

    def _build(self):
        f = ttk.Frame(self, padding=16); f.pack(fill='both', expand=True)
        ttk.Label(f, text='Video cần lồng tiếng:').pack(anchor='w')
        r1 = ttk.Frame(f); r1.pack(fill='x', pady=4)
        ttk.Entry(r1, textvariable=self.video).pack(side='left', fill='x', expand=True)
        ttk.Button(r1, text='Chọn...', command=self.pick_video).pack(side='left', padx=6)
        ttk.Label(f, text='Giọng mẫu Adam:').pack(anchor='w', pady=(8, 0))
        ttk.Combobox(f, textvariable=self.ref, values=list(REFS), state='readonly',
                     width=38).pack(anchor='w', pady=4)
        ttk.Label(f, text='Từ khóa giúp nghe đúng (tên sản phẩm, thương hiệu, cách nhau bằng dấu phẩy):').pack(anchor='w', pady=(8, 0))
        ttk.Entry(f, textvariable=self.keywords).pack(fill='x', pady=4)
        ttk.Label(f, text='Âm lượng tiếng gốc giữ lại (0 = tắt hẳn):').pack(anchor='w', pady=(8, 0))
        ttk.Scale(f, from_=0, to=0.5, variable=self.keep_orig).pack(fill='x', pady=4)
        ttk.Checkbutton(f, text='Tự động nhấn nhá cảm xúc (không cần sửa thủ công)',
                        variable=self.auto_emotion).pack(anchor='w', pady=4)
        ttk.Button(f, text='🎙️ BẮT ĐẦU LỒNG TIẾNG', command=self.start).pack(pady=12, fill='x')
        self.prog = ttk.Progressbar(f, mode='indeterminate'); self.prog.pack(fill='x')
        self.logw = tk.Text(f, height=15, state='disabled'); self.logw.pack(fill='both', expand=True, pady=8)
        ttk.Label(f, text='Miễn phí • Chạy offline hoàn toàn • Tự động nhấn nhá, không cần sửa thủ công',
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
            work = WORKROOT; os.makedirs(work, exist_ok=True)
            name = os.path.splitext(os.path.basename(v))[0]
            out = os.path.join(os.path.dirname(v), f'{name}_adam_dubbed.mp4')

            self.log('Bước 1/6: Tách audio...')
            wav = os.path.join(work, 'orig.wav'); extract_audio(v, wav)
            total = wav_duration(wav)

            self.log('Bước 2/6: Nghe và tách lời thoại (Whisper)...')
            kw = self.keywords.get().strip()
            segs = transcribe(wav, self.log, initial_prompt=kw or None)
            if not segs:
                self.log('Không nghe được lời thoại nào. Dừng.'); return
            self.log(f'Đã tách {len(segs)} câu thoại.')

            if self.auto_emotion.get():
                self.log('Bước 3/6: Tự động gắn nhấn nhá cảm xúc...')
                auto_tag_segments(segs, self.log)
            else:
                self.log('Bước 3/6: Bỏ qua nhấn nhá cảm xúc (đã tắt).')
                for s in segs:
                    s['tagged'] = s['text']

            self.log('Bước 4/6: Đăng ký giọng Adam (VieNeu-TTS v3 Turbo)...')
            enroll_adam(ref, self.log)

            self.log('Bước 5/6: Tạo giọng lồng tiếng...')
            clips = []
            for i, s in enumerate(segs):
                txt = s.get('tagged', s['text'])
                self.log(f'  Câu {i+1}/{len(segs)}: "{txt[:50]}..."')
                adam_wav = os.path.join(work, f'seg{i}_adam.wav')
                synth_adam(txt, adam_wav, self.log)
                fitted = fit_duration(adam_wav, s['end'] - s['start'] + 0.15, self.log)
                if fitted != adam_wav:
                    final = os.path.join(work, f'seg{i}_final.wav')
                    ensure_wav_16k_mono(fitted, final)
                    clips.append((s['start'], final))
                else:
                    clips.append((s['start'], adam_wav))

            self.log('Bước 6/6: Làm ấm giọng + dựng track + ghép vào video...')
            dub = os.path.join(work, 'dubbed.wav')
            build_track(clips, total, dub, log=self.log)
            warm_voice(dub, self.log)
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
