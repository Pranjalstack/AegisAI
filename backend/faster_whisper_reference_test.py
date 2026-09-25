from pathlib import Path

from faster_whisper import WhisperModel


AUDIO_DIR = Path("audio")
REFERENCE_FILE = AUDIO_DIR / "aegis_reference_16k.wav"
NEARFIELD_PATTERN = "nearfield_*.wav"

MODEL_SIZE = "base"


def latest_nearfield():
    files = sorted(
        AUDIO_DIR.glob(NEARFIELD_PATTERN)
    )

    return files[-1] if files else None


def transcribe(model, audio_path):
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    parts = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            parts.append(text)

    return " ".join(parts).strip(), info


def print_result(label, path, model):
    print(f"\n===== {label} =====")
    print(f"File: {path}")

    try:
        text, info = transcribe(
            model,
            path,
        )

        if not text:
            text = "[NO TRANSCRIPTION]"

        print(
            f"Detected language: "
            f"{info.language}"
        )

        print(
            f"Language probability: "
            f"{info.language_probability:.3f}"
        )

        print(
            f"Transcript:\n{text}"
        )

    except Exception as exc:
        print(
            f"Transcription failed: "
            f"{type(exc).__name__}: {exc}"
        )


def main():
    print(
        "===== AEGIS FASTER-WHISPER "
        "REFERENCE CONTROL TEST ====="
    )

    if not REFERENCE_FILE.exists():
        print(
            "\nReference file not found:"
        )
        print(REFERENCE_FILE)
        print(
            "\nRun "
            "backend.whisper_reference_test "
            "first."
        )
        return

    nearfield = latest_nearfield()

    print("\n===== LOADING FASTER-WHISPER =====")
    print(f"Model: {MODEL_SIZE}")
    print("Device: CPU")
    print("Compute type: int8")

    try:
        model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
    except Exception as exc:
        print(
            f"Model loading failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    print("Model loaded.")

    print_result(
        "KNOWN-GOOD REFERENCE",
        REFERENCE_FILE,
        model,
    )

    if nearfield:
        print_result(
            "REAL MICROPHONE",
            nearfield,
            model,
        )
    else:
        print(
            "\nNo near-field microphone recording found."
        )

    print("\n===== TEST COMPLETE =====")
    print(
        "\nExpected reference:"
    )
    print(
        "This is an Aegis microphone test."
    )


if __name__ == "__main__":
    main()