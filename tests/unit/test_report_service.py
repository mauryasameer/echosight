from src.services.eval_service import CaptionEvalResult
from src.services.report_service import build_report


def test_report_includes_captions_and_audio_tag():
    report = build_report(
        title="Test Report",
        image_path="/fake/img.jpg",
        raw_caption="dog running grass",
        enriched_description="A dog is running through the grass.",
        audio_b64="ZmFrZWF1ZGlv",
        attention_fig=None,
        eval_result=None,
        epochs_trained=5,
    )
    html = report.to_html()
    assert "dog running grass" in html
    assert "A dog is running through the grass." in html
    assert '<audio controls src="data:audio/mpeg;base64,ZmFrZWF1ZGlv">' in html
    assert "Run Info" in html
    assert "5" in html


def test_report_escapes_html_in_captions():
    report = build_report(
        title="Test Report",
        image_path="/fake/img.jpg",
        raw_caption="<script>alert(1)</script>",
        enriched_description="A dog is running.",
        audio_b64="ZmFrZWF1ZGlv",
        attention_fig=None,
        eval_result=None,
        epochs_trained=5,
    )
    html = report.to_html()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_report_includes_eval_scores_when_given():
    result = CaptionEvalResult(bleu4=0.42, mean_rouge_l=0.55)
    report = build_report(
        title="Test Report", image_path="/fake/img.jpg", raw_caption="a",
        enriched_description="b", audio_b64="ZmFrZQ==", attention_fig=None,
        eval_result=result, epochs_trained=5,
    )
    html = report.to_html()
    assert "bleu4" in html
    assert "mean_rouge_l" in html
