from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


AUDIO_DIR = Path("audio")
MODEL_NAME = "openai/whisper-base"


def latest_file(pattern):
    files = sorted(AUDIO_DIR.glob(pattern))
    return files[-1] if files else None


def load_whisper():
    print("===== LOADING LOCAL WHISPER =====")

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32

    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU mode")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )

    whisper = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )

    print("Whisper loaded.")
    return whisper


def transcribe(whisper, path):
    result = whisper(
        str(path),
        generate_kwargs={
            "task": "transcribe",
            "language": "en",
            "temperature": 0,
            "condition_on_prev_tokens": False,
        },
    )

    text = result.get("text", "").strip()

    return text if text else "[NO TRANSCRIPTION]"


def main():
    print("===== AEGIS CHANNEL TRANSCRIPTION TEST =====")

    files = {
        "LEFT": latest_file("mic_left_*.wav"),
        "RIGHT": latest_file("mic_right_*.wav"),
        "MONO": latest_file("mic_mono_*.wav"),
    }

    for label, path in files.items():
        print(f"\n{label}: {path}")

    available = {
        label: path
        for label, path in files.items()
        if path is not None
    }

    if not available:
        print("\nNo channel recordings found.")
        return

    whisper = load_whisper()

    print("\n===== TRANSCRIPTION RESULTS =====")

    for label, path in available.items():
        print(f"\n--- {label} ---")
        print(f"File: {path}")

        try:
            transcript = transcribe(
                whisper,
                path,
            )

            print(f"Transcript: {transcript}")

        except Exception as exc:
            print(
                f"Failed: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n===== TEST COMPLETE =====")
    print("\nExpected:")
    print("This is an Aegis microphone test.")


if __name__ == "__main__":
    main()