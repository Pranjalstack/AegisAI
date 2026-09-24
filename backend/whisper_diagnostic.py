# backend/whisper_diagnostic.py

from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)


MODEL_NAME = "openai/whisper-base"


def find_latest_chunk():
    audio_dir = Path("audio")

    chunks = list(
        audio_dir.glob("live_chunk_*.wav")
    )

    if not chunks:
        raise FileNotFoundError(
            "No live microphone recordings were found."
        )

    return max(
        chunks,
        key=lambda path: path.stat().st_mtime,
    )


def main():

    audio_path = find_latest_chunk()

    print(
        "\n===== AEGIS WHISPER DIAGNOSTIC ====="
    )

    print(
        f"Audio: {audio_path}"
    )

    print(
        "Loading local Whisper..."
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )

    device = (
        0
        if torch.cuda.is_available()
        else -1
    )

    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
    )

    print(
        "Transcribing with explicit English settings..."
    )

    result = recognizer(
        str(audio_path),
        generate_kwargs={
            "task": "transcribe",
            "language": "en",
            "temperature": 0.0,
            "num_beams": 5,
            "condition_on_prev_tokens": False,
        },
    )

    transcript = str(
        result.get(
            "text",
            ""
        )
    ).strip()

    print(
        "\n===== WHISPER RESULT ====="
    )

    print(
        "Transcript:"
    )

    if transcript:
        print(
            transcript
        )
    else:
        print(
            "[EMPTY]"
        )


if __name__ == "__main__":
    main()