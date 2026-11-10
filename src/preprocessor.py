"""Module tiền xử lý ngôn ngữ tự nhiên tiếng Việt 6 bước, băm SHA-256 và ẩn danh PII.

Đồ án tốt nghiệp: Hệ thống thu thập và phân tích cảm xúc từ ý kiến đánh giá của cộng đồng trong Đại học Thái Nguyên
Sinh viên: Nguyễn Văn Chiến - MSSV: DTC235200079 - Lớp: CN CNTT K22A (ICTU)
Giảng viên hướng dẫn: TS. Hà Thị Thanh
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Dict

try:
    from pyvi import ViTokenizer
except ImportError:  # pragma: no cover
    ViTokenizer = None  # type: ignore[assignment]

# Biểu thức chính quy nhận diện URL, thẻ HTML và thông tin định danh cá nhân (PII)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<(?!/?(?:EMAIL|PHONE|STUDENT_ID)\b)[^>]+>")
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE
)
_STUDENT_ID_RE = re.compile(r"\bDT[CEKNSYZFQU]\d{6,10}\b", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?84|0)(?:3[2-9]|5[25689]|7[06-9]|8[1-9]|9[0-9])(?:[\s.\-]?\d){7}(?!\d)"
)
_ELONGATED_CHAR_RE = re.compile(r"([a-zA-Zà-ỹÀ-Ỹ])\1{2,}", re.UNICODE)

# Token giữ chỗ tạm thời để bảo vệ <EMAIL>, <PHONE>, <STUDENT_ID> khi đi qua chữ thường và ViTokenizer
_PII_PLACEHOLDERS: Dict[str, str] = {
    "<EMAIL>": "phtokemail",
    "<PHONE>": "phtokphone",
    "<STUDENT_ID>": "phtokstudentid",
}
_REVERSE_PII_PLACEHOLDERS: Dict[str, str] = {
    v: k for k, v in _PII_PLACEHOLDERS.items()
}


class TNUStudentFeedbackPreprocessor:
    """Pipeline tiền xử lý 6 bước chuẩn hóa văn bản phản hồi tiếng Việt miền Đại học Thái Nguyên."""

    def __init__(self, secret_salt: str | None = None) -> None:
        # Không lộ Secret Salt cứng trên giao diện; đọc từ biến môi trường
        self.secret_salt: str = secret_salt or os.getenv("SECRET_SALT", "TNU_ICTU_2026")

        # Từ điển Emoji & Emoticon (bắt buộc sắp xếp theo độ dài khóa giảm dần khi thay thế)
        self.emoji_dict: Dict[str, str] = {
            ":))))": " tiêu_cực_mỉa_mai ",
            ":)))": " tiêu_cực_mỉa_mai ",
            ":))": " tích_cực_cười ",
            ":)": " tích_cực_vui ",
            ":-)))": " tiêu_cực_mỉa_mai ",
            ":-)": " tích_cực_vui ",
            ":((((": " tiêu_cực_rất_buồn ",
            ":(((": " tiêu_cực_rất_buồn ",
            ":((": " tiêu_cực_rất_buồn ",
            ":(": " tiêu_cực_buồn ",
            ":-(": " tiêu_cực_buồn ",
            "🥰": " tích_cực_yêu_thích ",
            "😍": " tích_cực_yêu_thích ",
            "❤️": " tích_cực_yêu_thích ",
            "❤": " tích_cực_yêu_thích ",
            "💖": " tích_cực_yêu_thích ",
            "👍": " tích_cực_tốt ",
            "👏": " tích_cực_khen_ngợi ",
            "🌟": " tích_cực_xuất_sắc ",
            "🔥": " tích_cực_ấn_tượng ",
            "😊": " tích_cực_hài_lòng ",
            "😁": " tích_cực_vui ",
            "😂": " trung_tính_cười ",
            "🤣": " trung_tính_cười ",
            "😭": " tiêu_cực_khóc ",
            "😢": " tiêu_cực_buồn ",
            "😞": " tiêu_cực_thất_vọng ",
            "😔": " tiêu_cực_buồn ",
            "😡": " tiêu_cực_tức_giận ",
            "🤬": " tiêu_cực_tức_giận ",
            "👎": " tiêu_cực_kém ",
            "😤": " tiêu_cực_bức_xúc ",
            "😴": " tiêu_cực_buồn_ngủ ",
            "🥱": " tiêu_cực_chán ",
        }

        # Từ điển giải mã viết tắt / teencode học đường miền ĐHTN
        self.teencode_dict: Dict[str, str] = {
            "gv": "giảng viên",
            "gviên": "giảng viên",
            "sv": "sinh viên",
            "sviên": "sinh viên",
            "hs": "học sinh",
            "ko": "không",
            "k": "không",
            "kh": "không",
            "hong": "không",
            "hok": "không",
            "dc": "được",
            "đc": "được",
            "dk": "đăng ký",
            "đk": "đăng ký",
            "đki": "đăng ký",
            "hỉu": "hiểu",
            "hiu": "hiểu",
            "iu": "yêu",
            "bik": "biết",
            "bit": "biết",
            "bt": "bình thường",
            "bth": "bình thường",
            "bthg": "bình thường",
            "ntn": "như thế nào",
            "j": "gì",
            "gi": "gì",
            "vs": "với",
            "mik": "mình",
            "mk": "mình",
            "m": "mình",
            "r": "rồi",
            "rùi": "rồi",
            "nc": "nói chuyện",
            "qt": "quá trình",
            "kt": "kiểm tra",
            "th": "thực hành",
            "lt": "lý thuyết",
            "tc": "tín chỉ",
            "hp": "học phần",
            "ctđt": "chương trình đào tạo",
            "csvc": "cơ sở vật chất",
            "ktx": "ký túc xá",
            "tv": "thư viện",
            "cntt": "công nghệ thông tin",
            "đhtn": "đại học thái nguyên",
            "ictu": "trường đại học công nghệ thông tin và truyền thông",
            "tnut": "trường đại học kỹ thuật công nghiệp",
            "tueba": "trường đại học kinh tế và quản trị kinh doanh",
            "tnue": "trường đại học sư phạm",
            "tuaf": "trường đại học nông lâm",
            "tump": "trường đại học y dược",
            "tnus": "trường đại học khoa học",
            "sfl": "trường ngoại ngữ",
            "istnu": "khoa quốc tế",
            "wá": "quá",
            "wa": "quá",
            "qá": "quá",
            "đag": "đang",
            "cx": "cũng",
            "lun": "luôn",
            "nv": "nhân viên",
            "cvht": "cố vấn học tập",
            "pđt": "phòng đào tạo",
            "ctsv": "công tác sinh viên",
        }

        # Bảng chuẩn hóa thanh điệu tiếng Việt (oà -> òa, uý -> úy...)
        self.tone_normalization_map: Dict[str, str] = {
            "oà": "òa",
            "oá": "óa",
            "oả": "ỏa",
            "oã": "õa",
            "oạ": "ọa",
            "oè": "òe",
            "oé": "óe",
            "oẻ": "ỏe",
            "oẽ": "õe",
            "oẹ": "ọe",
            "uỳ": "ùy",
            "uý": "úy",
            "uỷ": "ủy",
            "uỹ": "ũy",
            "uỵ": "ụy",
        }

    def redact_pii_only(self, text: str) -> str:
        """Bước 1: Chuẩn hóa Unicode NFC, xóa URL/HTML và ẩn danh PII theo Nghị định 13/2023/NĐ-CP."""
        if not isinstance(text, str) or not text.strip():
            return ""
        # 1. Chuẩn hóa Unicode dựng sẵn NFC
        cleaned = unicodedata.normalize("NFC", text)
        # 2. Xóa URL và thẻ HTML rác
        cleaned = _URL_RE.sub(" ", cleaned)
        cleaned = _HTML_TAG_RE.sub(" ", cleaned)
        # 3. Thay thế Email -> <EMAIL>, Mã SV ĐHTN -> <STUDENT_ID>, SĐT -> <PHONE>
        cleaned = _EMAIL_RE.sub("<EMAIL>", cleaned)
        cleaned = _STUDENT_ID_RE.sub("<STUDENT_ID>", cleaned)
        cleaned = _PHONE_RE.sub("<PHONE>", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _replace_emojis(self, text: str) -> str:
        """Bước 2: Sắp xếp khóa emoji/emoticon theo độ dài giảm dần trước khi thay thế."""
        out = text
        for emo_key in sorted(self.emoji_dict.keys(), key=len, reverse=True):
            if emo_key in out:
                out = out.replace(emo_key, self.emoji_dict[emo_key])
        return re.sub(r"\s+", " ", out).strip()

    def _normalize_elongated_and_lower(self, text: str) -> str:
        """Bước 3: Bảo vệ token PII, thu gọn ký tự lặp kéo dài (siêuuuu -> siêu) và chuyển chữ thường."""
        protected = text
        for pii_tok, placeholder in _PII_PLACEHOLDERS.items():
            protected = protected.replace(pii_tok, f" {placeholder} ")

        lowered = protected.lower()
        collapsed = _ELONGATED_CHAR_RE.sub(r"\1", lowered)
        # Giữ lại chữ cái tiếng Việt, chữ số, dấu gạch dưới (của emoji token) và dấu câu cơ bản
        collapsed = re.sub(
            r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ.,!?_]",
            " ",
            collapsed,
        )
        collapsed = re.sub(r"\s+", " ", collapsed).strip()
        return self._restore_pii_tokens(collapsed)

    def _expand_teencode_and_tone(self, text: str) -> str:
        """Bước 4 & 5: Giải mã teencode/viết tắt học đường ĐHTN và chuẩn hóa vị trí dấu thanh."""
        protected = text
        for pii_tok, placeholder in _PII_PLACEHOLDERS.items():
            protected = protected.replace(pii_tok, placeholder)

        # Tách dấu câu khỏi từ để khớp chính xác từ viết tắt
        spaced_punct = re.sub(r"([.,!?])", r" \1 ", protected)
        tokens = spaced_punct.split()
        expanded_tokens = [self.teencode_dict.get(tok, tok) for tok in tokens]
        out = " ".join(expanded_tokens)

        for old_tone, new_tone in self.tone_normalization_map.items():
            out = out.replace(old_tone, new_tone)

        out = re.sub(r"\s+", " ", out).strip()
        return self._restore_pii_tokens(out)

    def _segment_words(self, text: str) -> str:
        """Bước 6: Tách từ tiếng Việt bằng PyVi (ViTokenizer) và khôi phục nguyên vẹn token PII."""
        protected = text
        for pii_tok, placeholder in _PII_PLACEHOLDERS.items():
            protected = protected.replace(pii_tok, placeholder)

        if ViTokenizer is not None:
            segmented = ViTokenizer.tokenize(protected)
        else:
            segmented = protected

        segmented = re.sub(r"\s+", " ", segmented).strip()
        return self._restore_pii_tokens(segmented)

    @staticmethod
    def _restore_pii_tokens(text: str) -> str:
        restored = text
        for placeholder, pii_tok in _REVERSE_PII_PLACEHOLDERS.items():
            restored = re.sub(rf"\b{re.escape(placeholder)}\b", pii_tok, restored)
        return re.sub(r"\s+", " ", restored).strip()

    def preprocess_with_steps(self, raw_text: str) -> Dict[str, str]:
        """Trả về dict chứa kết quả tường minh của từng bước trong Pipeline NLP 6 bước."""
        step1 = self.redact_pii_only(raw_text)
        step2 = self._replace_emojis(step1)
        step3 = self._normalize_elongated_and_lower(step2)
        step4_5 = self._expand_teencode_and_tone(step3)
        step6 = self._segment_words(step4_5)
        return {
            "step1_nfc_pii": step1,
            "step2_emoji": step2,
            "step3_elongated": step3,
            "step4_5_teencode_tone": step4_5,
            "step6_segmented": step6,
        }

    def preprocess(self, raw_text: str) -> str:
        """Thực thi toàn bộ Pipeline 6 bước và trả về chuỗi tách từ cuối cùng đưa vào PhoBERT."""
        return self.preprocess_with_steps(raw_text)["step6_segmented"]

    def compute_content_hash(self, text: str) -> str:
        """Băm SHA-256 của nội dung đã ẩn danh PII để làm khóa chống trùng lặp (content_hash)."""
        normalized = self.redact_pii_only(text).lower().strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def compute_author_hash(self, author_identifier: str = "anonymous") -> str:
        """Băm ẩn danh một chiều SHA-256 kết hợp SECRET_SALT từ biến môi trường."""
        payload = f"{self.secret_salt}:{author_identifier.strip().lower()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

