from __future__ import annotations

import logging

from meerax.llm.base import LLMProvider
from meerax.llm.prompt import PromptTemplate

logger = logging.getLogger(__name__)

ENRICH_PROMPT = PromptTemplate(
    "An image captioning model generated this short caption: '{caption}'. "
    "In one natural, descriptive sentence, expand it into a fuller description "
    "suitable for reading aloud to someone who cannot see the image. "
    "Do not invent details the caption doesn't support — only elaborate on what's stated."
)


def enrich_caption(raw_caption: str, llm: LLMProvider) -> str:
    try:
        prompt = ENRICH_PROMPT.render(caption=raw_caption)
        return llm.generate(prompt, temperature=0.0).content
    except Exception:
        logger.exception("Caption enrichment failed for caption: %s", raw_caption)
        return raw_caption
