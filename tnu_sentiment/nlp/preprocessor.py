"""Pipeline tiền xử lý tiếng Việt tương thích với checkpoint PhoBERT hiện hữu."""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, Dict, List, Optional, Tuple

from tnu_sentiment.constants import PIPELINE_VERSION
from tnu_sentiment.domain.models import PreprocessResult, PreprocessStepTrace
from tnu_sentiment.nlp.pii_masker import mask_pii

# Placeholder nội bộ không có ký tự lặp liền kề và chỉ gồm chữ cái thường ASCII
# để đảm bảo các bước regex / rút gọn ký tự lặp / tách từ pyvi KHÔNG làm hỏng token PII.
_PII_PLACEHOLDERS: Dict[str, str] = {
    "<EMAIL>": "phtokemail",
    "<PHONE>": "phtokphone",
    "<STUDENT_ID>": "phtokstudentid",
}

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

# Bảng thay thế Emoji/Emoticon tương thích tập huấn luyện TNU_Sentiment_Dataset_Final.
# Lưu ý kỹ thuật quan trọng:
# 1. Luôn sắp xếp khóa dài trước khóa ngắn (ví dụ ':)))' trước ':))' trước ':)').
# 2. Không mặc định gán ':)))' thành 'tiêu cực mỉa mai' cho mọi câu, tránh bóp méo câu khen vui vẻ.
_EMOJI_MAP_RAW: Dict[str, str] = {
    "❤️": " tích cực yêu thích ",
    "❤": " tích cực yêu thích ",
    "🥰": " tích cực yêu thích ",
    "😍": " tích cực yêu thích ",
    "👍": " tích cực tốt ",
    "👏": " tích cực khen ngợi ",
    "😁": " tích cực vui vẻ ",
    "😊": " tích cực hài lòng ",
    "😢": " tiêu cực buồn ",
    "😭": " tiêu cực buồn khóc ",
    "👎": " tiêu cực kém ",
    "😡": " tiêu cực tức giận ",
    "🤬": " tiêu cực phẫn nộ ",
    "🥵": " tiêu cực nóng bức ",
    # Emoticon văn bản: khớp chuỗi dài trước chuỗi ngắn, giữ trung tính biểu cảm cười
    ":))))": " ",
    ":)))": " ",
    ":))": " ",
    ":)": " ",
    "=)))": " ",
    "=))": " ",
    ":(((": " tiêu cực buồn ",
    ":((": " tiêu cực buồn ",
    ":(": " tiêu cực buồn ",
}

_SORTED_EMOJI_ITEMS: List[Tuple[str, str]] = sorted(
    _EMOJI_MAP_RAW.items(), key=lambda kv: len(kv[0]), reverse=True
)

# Từ điển chuẩn hóa viết tắt & teencode khớp với dữ liệu huấn luyện checkpoint
_ABBREVIATION_MAP: Dict[str, str] = {
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hok": "không",
    "đc": "được",
    "dc": "được",
    "dk": "đăng ký",
    "đk": "đăng ký",
    "hỉu": "hiểu",
    "qá": "quá",
    "qa": "quá",
    "lun": "luôn",
    "xịn": "tốt",
    "sv": "sinh viên",
    "gv": "giảng viên",
    "ktx": "ký túc xá",
    "csht": "cơ sở hạ tầng",
    "csvc": "cơ sở vật chất",
    "đatn": "đồ án tốt nghiệp",
    "datn": "đồ án tốt nghiệp",
    "cthssv": "công tác học sinh sinh viên",
    "đhtn": "đại học thái nguyên",
    "dhtn": "đại học thái nguyên",
    "ictu": "trường đại học công nghệ thông tin và truyền thông",
    "dtc": "trường đại học công nghệ thông tin và truyền thông",
    "tnut": "trường đại học kỹ thuật công nghiệp",
    "dtk": "trường đại học kỹ thuật công nghiệp",
    "tueba": "trường đại học kinh tế và quản trị kinh doanh",
    "dte": "trường đại học kinh tế và quản trị kinh doanh",
    "tnue": "trường đại học sư phạm",
    "dts": "trường đại học sư phạm",
    "tuaf": "trường đại học nông lâm",
    "dtn": "trường đại học nông lâm",
    "tump": "trường đại học y dược",
    "dty": "trường đại học y dược",
    "tnus": "trường đại học khoa học",
    "dtz": "trường đại học khoa học",
    "sfl": "trường ngoại ngữ",
    "dtf": "trường ngoại ngữ",
    "istnu": "khoa quốc tế",
}

_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREVIATION_MAP, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

# Rút gọn ký tự lặp kéo dài từ 3 ký tự liên tiếp trở lên (ví dụ: siêuuuu -> siêu)
# mà không làm hỏng từ có 2 ký tự giống nhau hợp lệ (ví dụ: moodle, google, oop, cntt)
_ELONGATED_CHAR_RE = re.compile(r"([a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])\1{2,}")

# Giữ chữ cái tiếng Việt, chữ số, khoảng trắng và dấu câu cơ bản như tập huấn luyện
_ALLOWED_CHARS_RE = re.compile(
    r"[^a-z0-9\s,.\?!àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ_]"
)

# Từ nối tương phản gợi ý câu đa ý / nhiều khía cạnh
_MULTI_ASPECT_RE = re.compile(
    r"\b(?:nhưng|tuy\s+nhiên|mặc\s+dù|dù\s+vậy|mặt\s+khác|bù\s+lại|ngược\s+lại)\b",
    flags=re.IGNORECASE,
)


class SegmenterUnavailableError(RuntimeError):
    """Báo lỗi tường minh khi thiếu bộ tách từ pyvi thay vì âm thầm đổi segmenter."""


def _get_pyvi_segmenter() -> Tuple[Callable[[str], str], str]:
    try:
        from pyvi import ViTokenizer  # type: ignore

        return ViTokenizer.tokenize, "pyvi.ViTokenizer"
    except Exception as exc:  # pragma: no cover
        raise SegmenterUnavailableError(
            f"Không thể nạp bộ tách từ 'pyvi.ViTokenizer' bắt buộc cho checkpoint PhoBERT: {exc}"
        ) from exc


class VietnamesePreprocessor:
    """Thực thi tiền xử lý văn bản tiếng Việt và ghi nhận trace thực tế của từng bước."""

    def __init__(
        self,
        segmenter_fn: Optional[Callable[[str], str]] = None,
        segmenter_name: str = "pyvi.ViTokenizer",
        require_segmenter: bool = True,
    ) -> None:
        self._custom_segmenter = segmenter_fn
        self._segmenter_name = segmenter_name
        self._require_segmenter = require_segmenter

    def _resolve_segmenter(self) -> Tuple[Optional[Callable[[str], str]], str]:
        if self._custom_segmenter is not None:
            return self._custom_segmenter, self._segmenter_name
        try:
            return _get_pyvi_segmenter()
        except SegmenterUnavailableError:
            if self._require_segmenter:
                raise
            return None, "unavailable"

    def run(self, raw_text: str) -> PreprocessResult:
        if not isinstance(raw_text, str):
            raw_text = ""

        steps: List[PreprocessStepTrace] = []
        compat_notes: List[str] = []

        # Bước 1: Che thông tin cá nhân (PII) và loại bỏ URL
        masked_text, pii_detected = mask_pii(raw_text.strip())
        url_cleaned = _URL_RE.sub(" ", masked_text)
        url_cleaned = re.sub(r"\s+", " ", url_cleaned).strip()
        steps.append(
            PreprocessStepTrace(
                step_name="1. Che thông tin định danh (PII) & URL",
                description=(
                    f"Phát hiện PII: {', '.join(pii_detected)}"
                    if pii_detected
                    else "Không phát hiện email, SĐT hoặc mã sinh viên; loại bỏ URL nếu có."
                ),
                output_text=url_cleaned,
            )
        )

        # Bảo vệ các token <EMAIL>, <PHONE>, <STUDENT_ID> trước khi chuyển chữ thường & làm sạch
        protected = url_cleaned
        for token, placeholder in _PII_PLACEHOLDERS.items():
            protected = protected.replace(token, f" {placeholder} ")

        # Bước 2: Chuẩn hóa bảng mã Unicode dựng sẵn (NFC) và chữ thường
        normalized = unicodedata.normalize("NFC", protected).lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        steps.append(
            PreprocessStepTrace(
                step_name="2. Chuẩn hóa Unicode NFC & chữ thường",
                description="Đưa văn bản về bảng mã Unicode dựng sẵn (NFC) và chữ viết thường.",
                output_text=self._restore_pii_tokens(normalized),
            )
        )

        # Bước 3: Thay thế Emoji/Emoticon (khớp chuỗi dài trước chuỗi ngắn)
        emoji_text = normalized
        emoticon_smile_seen = any(sym in emoji_text for sym in (":))))", ":)))", ":))", ":)", "=)))", "=))"))
        for sym, replacement in _SORTED_EMOJI_ITEMS:
            if sym in emoji_text:
                emoji_text = emoji_text.replace(sym, replacement)
        emoji_text = re.sub(r"\s+", " ", emoji_text).strip()
        if emoticon_smile_seen:
            compat_notes.append(
                "Biểu cảm ':)))' / ':)' được khớp theo thứ tự chuỗi dài trước chuỗi ngắn và giữ trung tính "
                "(không ép thành 'tiêu_cực_mỉa_mai' như một số mẫu sinh tự động)."
            )
        if emoji_text != normalized:
            steps.append(
                PreprocessStepTrace(
                    step_name="3. Ánh xạ biểu tượng cảm xúc (Emoji/Emoticon)",
                    description="Chuyển đổi emoji sang cụm từ cảm xúc tương thích tập huấn luyện (khớp dài trước ngắn).",
                    output_text=self._restore_pii_tokens(emoji_text),
                )
            )

        # Bước 4: Rút gọn ký tự kéo dài (>=3) và giải nghĩa từ viết tắt / teencode
        collapsed = _ELONGATED_CHAR_RE.sub(r"\1", emoji_text)
        expanded = _ABBREV_PATTERN.sub(
            lambda m: _ABBREVIATION_MAP[m.group(0).lower()],
            collapsed,
        )
        expanded = re.sub(r"\s+", " ", expanded).strip()
        if expanded != emoji_text:
            steps.append(
                PreprocessStepTrace(
                    step_name="4. Chuẩn hóa ký tự kéo dài & từ viết tắt/teencode",
                    description="Rút gọn ký tự lặp kéo dài (>=3 ký tự) và mở rộng từ viết tắt học đường ĐHTN.",
                    output_text=self._restore_pii_tokens(expanded),
                )
            )

        # Bước 5: Làm sạch ký tự đặc biệt dư thừa (bảo toàn token PII)
        cleaned = _ALLOWED_CHARS_RE.sub(" ", expanded)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        steps.append(
            PreprocessStepTrace(
                step_name="5. Làm sạch ký tự nhiễu & chuẩn hóa khoảng trắng",
                description="Loại bỏ ký hiệu nhiễu ngoài bảng chữ cái tiếng Việt, chữ số và dấu câu cơ bản.",
                output_text=self._restore_pii_tokens(cleaned),
            )
        )

        # Bước 6: Tách từ tiếng Việt bằng pyvi.ViTokenizer (đúng segmenter của checkpoint)
        segmenter_fn, seg_name = self._resolve_segmenter()
        if segmenter_fn is not None and cleaned:
            segmented_raw = segmenter_fn(cleaned)
            final_text = self._restore_pii_tokens(segmented_raw)
            final_text = re.sub(r"\s+", " ", final_text).strip()
            steps.append(
                PreprocessStepTrace(
                    step_name=f"6. Tách từ tiếng Việt ({seg_name})",
                    description="Ghép âm tiết từ ghép bằng dấu gạch dưới '_' theo chuẩn đầu vào của PhoBERT.",
                    output_text=final_text,
                )
            )
        else:
            final_text = self._restore_pii_tokens(cleaned)

        is_multi_aspect = bool(_MULTI_ASPECT_RE.search(masked_text))

        return PreprocessResult(
            masked_content=masked_text,
            processed_content=final_text,
            pii_types_detected=pii_detected,
            is_multi_aspect=is_multi_aspect,
            steps=steps,
            pipeline_version=PIPELINE_VERSION,
            segmenter_name=seg_name,
            compatibility_notes=compat_notes,
        )

    @staticmethod
    def _restore_pii_tokens(text: str) -> str:
        restored = text
        for token, placeholder in _PII_PLACEHOLDERS.items():
            restored = re.sub(rf"\b{re.escape(placeholder)}\b", token, restored)
        return re.sub(r"\s+", " ", restored).strip()

