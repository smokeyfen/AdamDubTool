# AdamDubTool — Tool lồng tiếng AI giọng Adam

Tool miễn phí lồng giọng **Adam** (ElevenLabs) vào video tiếng Việt, khớp nhịp thoại, **tự động nhấn nhá cảm xúc**.

**Cách hoạt động (v7):** Whisper tách thoại → tự động gắn tag cảm xúc (`[vui vẻ]`, `[cười]`, `[hào hứng]`...) + chuẩn hoá dấu câu → VieNeu-TTS v3 Turbo đọc thẳng bằng giọng Adam (clone trực tiếp, không qua TTS trung gian) → EQ/compressor làm ấm giọng → FFmpeg ghép vào video.

- Chạy **offline hoàn toàn**, không cần mạng sau khi cài.
- **Không cần sửa lời thoại thủ công**: nhấn nhá được gắn tự động, có thể tắt trong giao diện.
- Miễn phí, không cần GPU.

## Tải bản chạy ngay (Windows)

- Link tải do Smokey cung cấp (file ZIP chia nhiều phần + script `ghep_file_v7.bat`, có thời hạn — hết hạn thì xin link mới).
- Tải đủ các phần + script về **cùng một thư mục**, chạy `ghep_file_v7.bat`, giải nén, chạy `AdamDubTool/AdamDubTool.exe`.
- Mỗi bản mới thay **toàn bộ thư mục cũ** (không trộn `_internal` cũ/mới).

## Chạy từ source (cho dev)

```bash
pip install -r requirements.txt
python src/app.py
```

Cần thêm: model VieNeu-TTS v3 Turbo đặt ở `build/models/vieneu`, model faster-whisper ở `build/models/`, file giọng mẫu ở `ref/`, `ffmpeg` trong PATH.

Build file exe cho Windows: xem `build/AdamDubTool.spec` (dùng PyInstaller chạy dưới Wine trên Linux).

## Cấu trúc source

- `src/app.py` — giao diện Tkinter (6 bước chạy)
- `src/pipeline.py` — tách audio, Whisper, VieNeu TTS, căn thời lượng, dựng track, mux FFmpeg
- `src/emotion.py` — **mới (v7)**: tự động gắn tag cảm xúc + chuẩn hoá dấu câu
- `src/postprocess.py` — **mới (v7)**: EQ + compressor + normalize làm ấm giọng (chỉ dùng numpy)

## Lưu ý

- Nếu có lỗi, file `adam_dub_error.log` nằm cạnh exe sẽ ghi chi tiết.
- Whisper mặc định dùng model `base` đóng gói sẵn; nếu có thư mục `models/faster-whisper-large-v3-turbo` đầy đủ (có `model.bin`) thì tool tự ưu tiên dùng.
