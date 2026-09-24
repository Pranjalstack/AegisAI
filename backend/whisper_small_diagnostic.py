# backend/whisper_small_diagnostic.py

from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)


MODEL_NAME = "openai/whisper-small"


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
        "\n===== AEGIS WHISPER SMALL DIAGNOSTIC ====="
    )

    print(
        f"Audio: {audio_path}"
    )

    print(
        "Loading local Whisper Small..."
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    use_cuda = torch.cuda.is_available()

    dtype = (
        torch.float16
        if use_cuda
        else torch.float32
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )

    if use_cuda:
        model = model.to("cuda")

    device = (
        0
        if use_cuda
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
        "Transcribing with local Whisper Small..."
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
        "\n===== WHISPER SMALL RESULT ====="
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