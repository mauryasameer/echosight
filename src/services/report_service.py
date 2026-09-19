from __future__ import annotations

import html
from datetime import UTC, datetime

from matplotlib.figure import Figure
from meerax.report.builder import ReportBuilder, ReportSection

from src.services.eval_service import CaptionEvalResult

GOVERNANCE_BANNER = (
    "GOVERNANCE NOTICE: This report supports a sighted-assistance use case; it is not a "
    "safety-critical navigation, obstacle-detection, or hazard-warning system. The "
    "enriched description is LLM-generated elaboration on the model's own caption, not "
    "an independently verified description of the image's actual contents."
)


def _run_info_html(epochs_trained: int, timestamp: str) -> str:
    return (
        f"<p class='meta'>Run Info &mdash; epochs trained: {epochs_trained} | "
        f"generated: {html.escape(timestamp)}</p>"
    )


def build_report(
    title: str,
    image_path: str,
    raw_caption: str,
    enriched_description: str,
    audio_b64: str,
    attention_fig: Figure | None,
    eval_result: CaptionEvalResult | None,
    epochs_trained: int,
) -> ReportBuilder:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    report = ReportBuilder(html.escape(title), subtitle=GOVERNANCE_BANNER)

    caption_content = (
        f"<p><strong>Model caption:</strong> {html.escape(raw_caption)}</p>"
        f"<p><strong>Enriched description:</strong> {html.escape(enriched_description)}</p>"
        f'<audio controls src="data:audio/mpeg;base64,{audio_b64}"></audio>'
    )
    report.add_section(
        ReportSection(
            title=html.escape(image_path),
            content=caption_content,
            figures=[attention_fig] if attention_fig is not None else [],
        )
    )

    if eval_result is not None:
        report.add_section(
            ReportSection(
                title="Evaluation",
                content="",
                metrics={"bleu4": eval_result.bleu4, "mean_rouge_l": eval_result.mean_rouge_l},
            )
        )

    report.add_section(ReportSection(title="Run Info", content=_run_info_html(epochs_trained, timestamp)))
    return report
