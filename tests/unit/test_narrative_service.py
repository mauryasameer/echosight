from meerax.llm.base import LLMResponse

from src.services.narrative_service import enrich_caption


class _StubLLM:
    def generate(self, prompt, system=None, **kwargs):
        assert kwargs.get("temperature") == 0.0
        return LLMResponse(
            content="A brown dog is running happily through a grassy field.",
            model="stub", input_tokens=1, output_tokens=1,
        )

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


class _FailingLLM:
    def generate(self, prompt, system=None, **kwargs):
        raise RuntimeError("boom")

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def test_enrich_caption_returns_llm_output():
    result = enrich_caption("dog running grass", _StubLLM())
    assert result == "A brown dog is running happily through a grassy field."


def test_enrich_caption_falls_back_to_raw_caption_on_failure():
    result = enrich_caption("dog running grass", _FailingLLM())
    assert result == "dog running grass"
