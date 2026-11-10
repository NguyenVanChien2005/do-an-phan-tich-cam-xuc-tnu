"""Module định tuyến lai (Hybrid Router): Gọi Gemini API (gemini-2.5-flash) khi confidence < 0.65
kèm cơ chế Fallback về bộ quy tắc từ khóa miền ĐHTN khi chưa cấu hình GEMINI_API_KEY.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

TOPIC_ID_TO_NAME: Dict[int, str] = {
    1: "Giảng viên & Phương pháp giảng dạy",
    2: "Chương trình đào tạo & Học phần",
    3: "Cơ sở vật chất & Hạ tầng CNTT",
    4: "Khác (Hành chính, Học phí, CTSV)",
}

SENTIMENT_ID_TO_TEXT: Dict[int, str] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

SENTIMENT_ID_TO_VI: Dict[int, str] = {
    0: "Tiêu cực (Negative)",
    1: "Trung tính (Neutral)",
    2: "Tích cực (Positive)",
}

# Bộ từ khóa phân loại 4 chủ đề miền Đại học Thái Nguyên
_TOPIC_KEYWORDS: Dict[int, Tuple[str, ...]] = {
    1: (
        "giảng viên",
        "giảng_viên",
        "giáo viên",
        "giáo_viên",
        "thầy",
        "cô",
        "giảng dạy",
        "giảng_dạy",
        "giảng bài",
        "truyền đạt",
        "truyền_đạt",
        "nhiệt tình",
        "nhiệt_tình",
        "tâm huyết",
        "tâm_huyết",
        "chấm điểm",
        "hướng dẫn",
        "hướng_dẫn",
    ),
    2: (
        "học phần",
        "học_phần",
        "môn học",
        "môn_học",
        "tín chỉ",
        "tín_chỉ",
        "chương trình",
        "chương_trình",
        "giáo trình",
        "giáo_trình",
        "tài liệu",
        "tài_liệu",
        "lịch thi",
        "lịch học",
        "đề thi",
        "kiến thức",
        "kiến_thức",
        "thực hành",
        "thực_hành",
        "lý thuyết",
        "lý_thuyết",
    ),
    3: (
        "phòng học",
        "phòng_học",
        "phòng máy",
        "máy chiếu",
        "máy_chiếu",
        "điều hòa",
        "điều_hòa",
        "wifi",
        "mạng",
        "cơ sở vật chất",
        "cơ_sở_vật_chất",
        "ký túc xá",
        "thư viện",
        "thư_viện",
        "bàn ghế",
        "nhà vệ sinh",
        "bãi gửi xe",
        "cổng thông tin",
        "đăng ký tín chỉ",
        "hệ thống",
        "máy tính",
    ),
    4: (
        "học phí",
        "học_phí",
        "học bổng",
        "học_bổng",
        "cố vấn học tập",
        "phòng đào tạo",
        "công tác sinh viên",
        "thủ tục",
        "giấy xác nhận",
        "bảo hiểm",
        "hoạt động",
        "đoàn hội",
    ),
}

# Bộ từ khóa phát hiện câu mỉa mai / đa cực tính khi fallback không có Gemini API Key
_SARCASM_CUES: Tuple[str, ...] = (
    "tiêu_cực_mỉa_mai",
    ":)))",
    "mượt ghê",
    "tốt ghê",
    "hay ghê",
    "nhanh ghê",
    "quay mòng mòng",
    "sập liên tục",
)

_CONTRAST_CONJUNCTIONS: Tuple[str, ...] = (
    " nhưng ",
    " tuy nhiên ",
    " mặc dù ",
    " dù ",
    " song ",
)


def infer_topic_by_keywords(text: str) -> Tuple[int, str, str]:
    """Phân loại chủ đề dựa trên từ điển từ khóa học đường ĐHTN.

    Returns:
        Tuple (topic_id, topic_name, router_source)
    """
    lowered = (text or "").lower()
    scores: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    for tid, kw_list in _TOPIC_KEYWORDS.items():
        for kw in kw_list:
            if kw in lowered:
                scores[tid] += 2 if " " in kw or "_" in kw else 1

    best_tid = max(scores, key=lambda k: scores[k])
    if scores[best_tid] == 0:
        return 4, TOPIC_ID_TO_NAME[4], "Keyword Fallback Router"
    return best_tid, TOPIC_ID_TO_NAME[best_tid], "Keyword Fallback Router"


def is_gemini_configured() -> bool:
    """Kiểm tra biến môi trường GEMINI_API_KEY đã được cấu hình hợp lệ hay chưa."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(key and key not in ("your_gemini_api_key_here", "CHANGE_ME"))


def call_gemini_LowConfidence_router(
    masked_text: str,
    phobert_label: int,
    phobert_confidence: float,
    timeout_sec: int = 8,
) -> Dict[str, Any]:
    """Định tuyến sang Gemini API (gemini-2.5-flash) khi PhoBERT có confidence < 0.65.
    Nếu chưa cấu hình GEMINI_API_KEY hoặc lỗi mạng, tự động fallback về bộ quy tắc từ khóa.
    """
    topic_id, topic_name, _ = infer_topic_by_keywords(masked_text)
    lowered = f" {masked_text.lower()} "
    has_sarcasm = any(cue in lowered for cue in _SARCASM_CUES)
    has_contrast = any(conj in lowered for conj in _CONTRAST_CONJUNCTIONS)

    if is_gemini_configured():
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={api_key}"
        )
        prompt = (
            "Bạn là chuyên gia phân tích cảm xúc phản hồi sinh viên Đại học Thái Nguyên. "
            "Phân loại câu sau (đã che PII) vào 1 trong 3 nhãn cảm xúc (0: Negative, 1: Neutral, 2: Positive) "
            "và 1 trong 4 mã chủ đề (1: Giảng viên, 2: Chương trình đào tạo, 3: Cơ sở vật chất, 4: Khác). "
            "Trả về đúng JSON: {\"sentiment_label\": int, \"topic_id\": int, \"explanation\": str}.\n"
            f"Câu phản hồi: {masked_text}"
        )
        payload = json.dumps(
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw_resp = json.loads(resp.read().decode("utf-8"))
                candidates = raw_resp.get("candidates") or []
                text_part = (
                    candidates[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "{}")
                )
                parsed = json.loads(text_part)
                s_lbl = int(parsed.get("sentiment_label", phobert_label))
                t_id = int(parsed.get("topic_id", topic_id))
                if s_lbl not in (0, 1, 2):
                    s_lbl = phobert_label
                if t_id not in (1, 2, 3, 4):
                    t_id = topic_id
                return {
                    "router_mode": f"Gemini API ({model_name})",
                    "suggested_sentiment_label": s_lbl,
                    "suggested_sentiment_text": SENTIMENT_ID_TO_TEXT[s_lbl],
                    "suggested_topic_id": t_id,
                    "suggested_topic_name": TOPIC_ID_TO_NAME[t_id],
                    "review_status": "NEEDS_REVIEW",
                    "explanation": str(
                        parsed.get(
                            "explanation",
                            "Phân giải câu độ tự tin thấp qua Gemini 2.5 Flash.",
                        )
                    ),
                }
        except Exception as exc:
            fallback_reason = f"Gemini API không phản hồi ({type(exc).__name__}), chuyển sang Keyword Fallback."
    else:
        fallback_reason = "Chưa cấu hình GEMINI_API_KEY — sử dụng Keyword Rule Fallback."

    # Fallback phân giải bằng từ điển từ khóa (Keyword Heuristic)
    suggested_label = phobert_label
    notes = [fallback_reason]
    if has_sarcasm:
        suggested_label = 0
        notes.append("Phát hiện dấu hiệu mỉa mai/khen ngược (':)))' / 'mượt ghê').")
    if has_contrast:
        notes.append(
            "Câu ghép đa cực tính chứa liên từ tương phản ('nhưng'/'tuy nhiên') — gắn cờ NEEDS_REVIEW để kiểm duyệt."
        )

    return {
        "router_mode": "Keyword Rule Fallback",
        "suggested_sentiment_label": suggested_label,
        "suggested_sentiment_text": SENTIMENT_ID_TO_TEXT[suggested_label],
        "suggested_topic_id": topic_id,
        "suggested_topic_name": topic_name,
        "review_status": "NEEDS_REVIEW" if (phobert_confidence < 0.65 or has_contrast) else "AUTO_APPROVED",
        "explanation": " ".join(notes),
    }

