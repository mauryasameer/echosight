from meerax.llm.base import LLMResponse

from src.services.eval_service import evaluate_captions
from src.services.narrative_service import enrich_caption
from src.services.report_service import build_report
from src.services.tts_service import text_to_speech_b64


class _StubLLM:
    def generate(self, prompt, system=None, **kwargs):
        return LLMResponse(
            content="A small brown dog is running through a green field.",
            model="stub", input_tokens=1, output_tokens=1,
        )

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def test_full_pipeline_produces_nonempty_report(tmp_path, monkeypatch):
    # Exercise caption -> enrich -> TTS -> eval -> report end to end with a synthetic
    # caption (not a real trained checkpoint or real image) and a stubbed LLM.
    raw_caption = "dog running grass"
    enriched = enrich_caption(raw_caption, _StubLLM())
    assert enriched == "A small brown dog is running through a green field."

    audio_b64 = text_to_speech_b64(enriched)
    assert len(audio_b64) > 50

    eval_result = evaluate_captions(
        references=["a dog runs through the grass"], hypotheses=[raw_caption]
    )

    report = build_report(
        title="EchoSight Test Report",
        image_path="test_image.jpg",
        raw_caption=raw_caption,
        enriched_description=enriched,
        audio_b64=audio_b64,
        attention_fig=None,
        eval_result=eval_result,
        epochs_trained=1,
    )

    output_path = tmp_path / "report.html"
    report.save(str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
