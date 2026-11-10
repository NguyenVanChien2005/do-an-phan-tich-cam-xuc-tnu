"""Dịch vụ kiểm tra tính tương thích đối sánh và phân nhóm kết quả đánh giá mô hình."""

from __future__ import annotations

from typing import Dict, List, Tuple

from tnu_sentiment.domain.models import BenchmarkRun


class BenchmarkService:
    """Đảm bảo nguyên tắc học thuật:
    - Tách riêng tập Validation và Test.
    - Chỉ cho phép so sánh trực tiếp các run cùng tác vụ (task), cùng split,
      cùng evaluation_set_id (dấu vết tập mẫu) và cùng metric_definition_version.
    - Cùng n_samples nhưng khác evaluation_set_id KHÔNG được phép gộp so sánh trực tiếp.
    """

    @staticmethod
    def are_runs_comparable(run_a: BenchmarkRun, run_b: BenchmarkRun) -> Tuple[bool, str]:
        if run_a.task.upper() != run_b.task.upper():
            return (
                False,
                f"Khác tác vụ ({run_a.task} vs {run_b.task}) — không thể so sánh trực tiếp.",
            )
        if run_a.split.lower() != run_b.split.lower():
            return (
                False,
                f"Khác phân tập đánh giá ({run_a.split} vs {run_b.split}) — không trộn Validation và Test.",
            )
        if not run_a.evaluation_set_id or not run_b.evaluation_set_id:
            return (
                False,
                "Thiếu định danh tập mẫu (evaluation_set_id) để xác nhận cùng tập đánh giá.",
            )
        if run_a.evaluation_set_id != run_b.evaluation_set_id:
            if run_a.n_samples == run_b.n_samples:
                return (
                    False,
                    f"Cùng số lượng mẫu ({run_a.n_samples}) nhưng khác mã định danh tập đánh giá "
                    f"({run_a.evaluation_set_id} != {run_b.evaluation_set_id}) — không đủ bằng chứng cùng tập.",
                )
            return (
                False,
                f"Khác tập đánh giá ({run_a.evaluation_set_id}, n={run_a.n_samples} vs "
                f"{run_b.evaluation_set_id}, n={run_b.n_samples}).",
            )
        if run_a.metric_definition_version != run_b.metric_definition_version:
            return (
                False,
                f"Khác phiên bản định nghĩa metric ({run_a.metric_definition_version} vs "
                f"{run_b.metric_definition_version}).",
            )
        return True, "Cùng tác vụ, cùng tập đánh giá đã xác thực và cùng định nghĩa metric."

    @staticmethod
    def group_runs_by_task_and_split(
        runs: List[BenchmarkRun],
    ) -> Dict[Tuple[str, str], List[BenchmarkRun]]:
        grouped: Dict[Tuple[str, str], List[BenchmarkRun]] = {}
        for r in runs:
            key = (r.task.upper(), r.split)
            grouped.setdefault(key, []).append(r)
        return grouped

    @staticmethod
    def filter_comparable_group(
        anchor: BenchmarkRun, candidates: List[BenchmarkRun]
    ) -> Tuple[List[BenchmarkRun], List[Tuple[BenchmarkRun, str]]]:
        comparable: List[BenchmarkRun] = [anchor]
        excluded: List[Tuple[BenchmarkRun, str]] = []
        for c in candidates:
            if c.run_id == anchor.run_id:
                continue
            ok, reason = BenchmarkService.are_runs_comparable(anchor, c)
            if ok:
                comparable.append(c)
            else:
                excluded.append((c, reason))
        return comparable, excluded

