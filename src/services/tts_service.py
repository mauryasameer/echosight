from __future__ import annotations

import base64
import io

from gtts import gTTS


def text_to_speech_bytes(text: str, lang: str = "en") -> bytes:
    buffer = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buffer)
    return buffer.getvalue()


def text_to_speech_b64(text: str, lang: str = "en") -> str:
    return base64.b64encode(text_to_speech_bytes(text, lang=lang)).decode("ascii")
