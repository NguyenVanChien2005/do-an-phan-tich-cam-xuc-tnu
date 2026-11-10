"""Tích hợp LLM hỗ trợ tùy chọn: tắt mặc định, che PII trước khi gửi, kiểm tra schema đầu ra."""

from __future__ import annotations

import json
from typing import Optional

import httpx

from tnu_sentiment.config import AppConfig
from tnu_sentiment.domain.models import ComponentHealth, LLMSuggestion
from tnu_sentiment.models.topic_adapter import VALID_TOPIC_CODES
from tnu_sentiment.nlp.pii_masker import mask_pii


class OptionalLLMHelper:
    """Bộ hỗ trợ phân tích gợi ý qua API tương thích OpenAI Chat Completions (tùy chọn).
    - Không bao giờ tự gọi API nếu LLM_ENABLED=false hoặc người dùng không bấm yêu cầu.
    - Luôn che PII trước khi gửi văn bản ra bên ngoài.
    - Gợi ý của LLM chỉ mang tính tham khảo (NEEDS_REVIEW), không bao giờ tự thành VERIFIED.
    """

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config

    @property
    def is_enabled_and_configured(self) -> bool:
        return bool(
            self._cfg.llm_enabled
            and self._cfg.llm_api_base_url
            and self._cfg.llm_api_key
            and self._cfg.llm_model_name
        )

    def health(self) -> ComponentHealth:
        if not self._cfg.llm_enabled:
            return ComponentHealth(
                ready=False,
                user_badge="LLM hỗ trợ: Đang tắt (Tùy chọn)",
                technical_detail="LLM_ENABLED=false. Ứng dụng chính vận hành độc lập không cần LLM.",
            )
        if not self.is_enabled_and_configured:
            return ComponentHealth(
                ready=False,
                user_badge="LLM hỗ trợ: Thiếu cấu hình",
                technical_detail="Đã bật LLM_ENABLED nhưng thiếu LLM_API_BASE_URL, LLM_API_KEY hoặc LLM_MODEL_NAME.",
            )
        return ComponentHealth(
            ready=True,
            user_badge="LLM hỗ trợ: Sẵn sàng (Theo yêu cầu)",
            technical_detail=(
                f"Model={self._cfg.llm_model_name}, timeout={self._cfg.llm_timeout_seconds}s, "
                "bắt buộc che PII trước khi gửi."
            ),
        )

    def request_suggestion(self, raw_or_masked_text: str) -> LLMSuggestion:
        if not self.is_enabled_and_configured:
            raise RuntimeError("Chức năng LLM hỗ trợ chưa được bật và cấu hình đầy đủ.")

        safe_text, _ = mask_pii(raw_or_masked_text)
        endpoint = self._cfg.llm_api_base_url.rstrip("/") + "/chat/completions"

        system_prompt = (
            "Bạn là trợ lý hỗ trợ kiểm duyệt phản hồi giáo dục đại học. "
            "Chỉ trả về JSON hợp lệ theo đúng cấu trúc: "
            '{"sentiment_label": 0|1|2, "topic_code": "LECTURER"|"CURRICULUM"|"FACILITY"|"OTHER_SUPPORT"|"UNCLASSIFIED", '
            '"explanation": "giải thích ngắn tiếng Việt dưới 60 từ"}. '
            "Quy ước: 0=Tiêu cực, 1=Trung tính, 2=Tích cực."
        )

        payload = {
            "model": self._cfg.llm_model_name,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Phân tích phản hồi đã che PII sau:\n{safe_text}"},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._cfg.llm_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=float(self._cfg.llm_timeout_seconds)) as client:
            resp = client.post(endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()

        content = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Đầu ra LLM không phải đối tượng JSON.")

        raw_sent = parsed.get("sentiment_label")
        sent_label: Optional[int] = int(raw_sent) if raw_sent in (0, 1, 2) else None
        if sent_label not in (0, 1, 2):
            raise ValueError(f"sentiment_label từ LLM không hợp lệ: {raw_sent}")

        raw_topic = str(parsed.get("topic_code", "UNCLASSIFIED")).strip().upper()
        if raw_topic not in VALID_TOPIC_CODES:
            raise ValueError(f"topic_code từ LLM không thuộc danh mục hợp lệ: {raw_topic}")

        explanation = str(parsed.get("explanation", "")).strip()
        if not explanation:
            raise ValueError("Thiếu trường explanation từ LLM.")
        safe_explanation, _ = mask_pii(explanation[:300])

        return LLMSuggestion(
            suggested_sentiment_label=sent_label,
            suggested_topic_code=raw_topic,
            explanation=safe_explanation,
            model_name=self._cfg.llm_model_name,
        )

