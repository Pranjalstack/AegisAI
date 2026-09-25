from pathlib import Path

from faster_whisper import WhisperModel


AUDIO_DIR = Path("audio")
MODEL_SIZE = "small.en"


def latest_nearfield_recording():
    files = sorted(
        AUDIO_DIR.glob("nearfield_*.wav")
    )

    if not files:
        return None

    # Only use the original near-field recording.
    # Ignore converted or processed copies.
    original_files = [
        file
        for file in files
        if "_16k_mono" not in file.stem
        and "_clean" not in file.stem
        and "_trimmed" not in file.stem
        and "_enhanced" not in file.stem
    ]

    if original_files:
        return original_files[-1]

    return files[-1]


def load_model():
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

        print("Model loaded.")
        return model

    except Exception as exc:
        print(
            f"\nModel loading failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return None


def transcribe(model, audio_path):
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        temperature=0,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    parts = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            print(
                f"[{segment.start:.2f}s -> "
                f"{segment.end:.2f}s] {text}"
            )

            parts.append(text)

    transcript = " ".join(parts).strip()

    if not transcript:
        transcript = "[NO TRANSCRIPTION]"

    return transcript, info


def main():
    print("===== AEGIS FASTER-WHISPER TEST =====")

    audio_path = latest_nearfield_recording()

    if audio_path is None:
        print(
            "\nNo near-field recording found."
        )
        print(
            "Expected:"
            "\naudio\\nearfield_*.wav"
        )
        return

    print("\nAudio:")
    print(audio_path)

    model = load_model()

    if model is None:
        return

    print("\n===== TRANSCRIPTION =====")

    try:
        transcript, info = transcribe(
            model,
            audio_path,
        )

        print(
            f"\nDetected language: "
            f"{info.language}"
        )

        print(
            f"Language probability: "
            f"{info.language_probability:.3f}"
        )

        print("\n===== FINAL TRANSCRIPT =====")
        print(transcript)

    except Exception as exc:
        print(
            f"\nTranscription failed: "
            f"{type(exc).__name__}: {exc}"
        )

    print("\n===== TEST COMPLETE =====")

    print("\nExpected:")
    print("This is an Aegis microphone test.")


if __name__ == "__main__":
    main()