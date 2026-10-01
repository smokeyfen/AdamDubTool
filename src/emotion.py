"""v7 - Tự động gắn tag cảm xúc + chuẩn hoá dấu câu cho transcript.

Whisper cho ra text trần (không dấu cảm xúc). Module này phân tích ngữ nghĩa
tiếng Việt của từng câu và tự gắn tag cảm xúc VieNeu (đã kiểm chứng an toàn)
vào đầu câu + chuẩn hoá dấu câu cuối câu, để giọng Adam đọc có nhấn nhá
tự nhiên mà user không phải sửa thủ công.

Nguyên tắc an toàn:
- Chỉ dùng tag trong WHITELIST (đã test thực tế: sinh audio thành công,
  giọng không bị lệch - similarity giữa các phiên bản >= 0.89).
- Mỗi câu tối đa 1 tag, gắn ở đầu câu.
- Không đổi từ ngữ gốc, chỉ thêm tag + dấu câu.
"""

import re

# Tag đã kiểm chứng trên VieNeu-TTS v3 Turbo (giọng Adam giữ nguyên).
WHITELIST = {"vui vẻ", "hào hứng", "cười", "thở dài", "ngạc nhiên",
             "trầm ngâm", "thuyết phục", "tự tin", "dịu dàng", "bí mật"}

def _has(text, *words):
    t = text.lower()
    return any(w in t for w in words)

# ---------- bộ luật nhận diện cảm xúc (tiếng Việt, video bán hàng) ----------
def _detect_tag(text, is_first, is_last):
    t = text.strip()

    # 1. Cười / đùa
    if _has(t, "haha", "ha ha", "buồn cười", "hài hước", "đùa thôi", "cười"):
        return "cười"
    # 2. Ngạc nhiên / hook gây sốc
    if _has(t, "không thể tin", "tin được không", "sốc", "bất ngờ quá",
            "trời ơi", "ôi trời", "wow"):
        return "ngạc nhiên"
    # 3. Hào hứng: tính từ mạnh, câu cảm thán về sản phẩm
    if _has(t, "tuyệt vời", "xuất sắc", "hoàn hảo", "cực kỳ", "siêu ",
            "siêu phẩm", "quá đã", "quá đỉnh", "đỉnh cao", "khủng",
            "tuyệt đỉnh", "mê ly", "phê"):
        return "hào hứng"
    # 4. Vui vẻ: cảm xúc tích cực chung
    if _has(t, "vui", "hạnh phúc", "thích thú", "yêu thích", "tuyệt",
            "rất tốt", "rất hay"):
        return "vui vẻ"
    # 5. Thở dài / tiếc nuối
    if _has(t, "tiếc rằng", "đáng tiếc", "giá như", "phải chi"):
        return "thở dài"
    # 6. Trầm ngâm: vấn đề, khó khăn, băn khoăn
    if _has(t, "vấn đề là", "khó khăn", "băn khoăn", "lo lắng",
            "nhưng mà", "tuy nhiên"):
        return "trầm ngâm"
    # 7. Thuyết phục / CTA: kêu gọi hành động
    if _has(t, "mua ngay", "đặt hàng", "inbox", "liên hệ", "nhanh tay",
            "số lượng có hạn", "đừng bỏ lỡ", "chốt đơn", "nhắn tin ngay",
            "đăng ký ngay") or (is_last and _has(t, "cảm ơn", "hẹn gặp")):
        return "thuyết phục"
    # 8. Tự tin: cam kết, đảm bảo
    if _has(t, "cam kết", "đảm bảo", "chính hãng", "uy tín",
            "100%", "chắc chắn"):
        return "tự tin"
    # 9. Bí mật: bật mí, tiết lộ
    if _has(t, "bí mật", "bật mí", "tiết lộ", "mách nhỏ"):
        return "bí mật"
    # 10. Dịu dàng: cảm ơn, tình cảm, mẹ & bé
    if _has(t, "cảm ơn", "biết ơn", "yêu thương", "dịu nhẹ", "an toàn cho bé",
            "cho con"):
        return "dịu dàng"
    # 11. Câu mở đầu (hook) chưa có cảm xúc -> hào hứng để níu người xem
    if is_first:
        return "hào hứng"
    return None


_CTA_WORDS = ("mua ngay", "đặt hàng", "inbox", "liên hệ", "nhanh tay",
               "chốt đơn", "nhắn tin ngay", "đăng ký ngay", "số lượng có hạn",
               "đừng bỏ lỡ")
_EXCITED_WORDS = ("tuyệt vời", "xuất sắc", "quá đã", "trời ơi", "wow",
                  "tuyệt đỉnh", "siêu phẩm", "cực kỳ")


def _fix_punctuation(text):
    """Chuẩn hoá dấu câu cuối câu để VieNeu ngắt nhịp đúng."""
    t = text.strip()
    # "..." dở dang -> ","
    t = re.sub(r"\.{2,}\s*$", ",", t)
    cta = _has(t, *_CTA_WORDS)
    excited = _has(t, *_EXCITED_WORDS)
    # Câu hỏi thiếu dấu ? (nhưng CTA/cảm thán thì ưu tiên dấu !)
    if (not cta and not excited
            and re.search(r"(không|chưa|à|hả|nhé|nhỉ|đúng không|phải không|được không)\s*[.]?\s*$",
                          t, re.IGNORECASE) and not t.endswith("?")):
        t = re.sub(r"[.\s]*$", "?", t)
    # CTA / câu cảm thán -> dấu !
    elif (cta or excited) and not t.endswith("!"):
        t = re.sub(r"[.?\s]*$", "!", t)
    # Câu không có dấu kết -> thêm dấu chấm
    elif not t.endswith((".", "!", "?", ",", ":", ";")):
        t = t + "."
    return t


_TAG_RE = re.compile(r"^\s*\[[^\]]{1,30}\]\s*")

def tag_segment(text, is_first=False, is_last=False):
    """Trả về text đã gắn tag cảm xúc + chuẩn hoá dấu câu."""
    t = _TAG_RE.sub("", text.strip())  # gỡ tag cũ nếu có
    if not t:
        return t
    t = _fix_punctuation(t)
    tag = _detect_tag(t, is_first, is_last)
    if tag and tag in WHITELIST:
        return f"[{tag}] {t}"
    return t


def auto_tag_segments(segments, log=print):
    """segments: [{'start','end','text'}] -> thêm field 'tagged' cho mỗi segment."""
    n = len(segments)
    tagged_count = 0
    for i, s in enumerate(segments):
        s["tagged"] = tag_segment(s["text"], is_first=(i == 0), is_last=(i == n - 1))
        if s["tagged"] != s["text"]:
            tagged_count += 1
            log(f'  Cảm xúc câu {i+1}: "{s["tagged"][:60]}..."')
    log(f"Đã tự động gắn nhấn nhá cho {tagged_count}/{n} câu.")
    return segments
