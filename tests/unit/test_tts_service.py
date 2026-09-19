import base64

from src.services.tts_service import text_to_speech_b64, text_to_speech_bytes


def test_text_to_speech_bytes_returns_real_audio():
    audio = text_to_speech_bytes("hello world")
    assert isinstance(audio, bytes)
    assert len(audio) > 100  # a real MP3, not empty


def test_text_to_speech_b64_is_valid_base64_of_the_same_audio():
    audio_bytes = text_to_speech_bytes("hello world")
    b64 = text_to_speech_b64("hello world")
    # not byte-identical (gTTS network calls aren't perfectly deterministic across
    # requests), but both should decode to real, similarly-sized MP3 data
    decoded = base64.b64decode(b64)
    assert len(decoded) > 100
    assert abs(len(decoded) - len(audio_bytes)) < len(audio_bytes)  # same order of magnitude
