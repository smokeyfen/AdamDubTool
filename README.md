# AdamDubTool — Tool lồng tiếng AI giọng Adam

Tool miễn phí lồng giọng **Adam** (ElevenLabs) vào video tiếng Việt, khớp nhịp thoại.

**Cách hoạt động:** Whisper tách thoại → edge-tts tạo giọng đọc → OpenVoice đổi sang giọng Adam → FFmpeg ghép vào video.

## Tải bản chạy ngay (Windows)

Không cần cài Python. Bạn cần có **thư mục tool đầy đủ** (gồm `_internal`, `models`, `ref`...), rồi chỉ thay file `AdamDubTool.exe`:

- Tải `AdamDubTool.exe` mới về, copy đè vào thư mục `AdamDubTool` (nằm cạnh thư mục `_internal`).
- Link tải do Smokey cung cấp (có thời hạn — hết hạn thì xin link mới).

## Chạy từ source (cho dev)

```bash
pip install -r requirements.txt
python src/app.py
```

Build file exe cho Windows: xem `build/AdamDubTool.spec` (dùng PyInstaller chạy dưới Wine trên Linux).

## Lưu ý

- Lần chạy đầu cần mạng (tải model Whisper + gọi TTS).
- Nếu có lỗi, file `adam_dub_error.log` nằm cạnh exe sẽ ghi chi tiết.
