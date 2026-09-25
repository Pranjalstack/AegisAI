from pathlib import Path

from faster_whisper import WhisperModel


AUDIO_DIR = Path("audio")
MODEL_SIZE = "base"


def latest_nearfield_recording():
    files = sorted(
        AUDIO_DIR.glob("nearfield_*.wav")
    )

    return files[-1] if files else None


def main():
    print("===== AEGIS FASTER-WHISPER TEST =====")

    audio_path = latest_nearfield_recording()

    if audio_path is None:
        print("\nNo near-field recording found.")
        print("Expected: audio\\nearfield_*.wav")
        return

    print(f"\nAudio:")
    print(audio_path)

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

    except Exception as exc:
        print(
            f"Model loading failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    print("\n===== TRANSCRIPTION =====")

    try:
        segments, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        print(
            f"Detected language: "
            f"{info.language}"
        )

        print(
            f"Language probability: "
            f"{info.language_probability:.3f}"
        )

        transcript_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                transcript_parts.append(text)

                print(
                    f"[{segment.start:.2f}s -> "
                    f"{segment.end:.2f}s] "
                    f"{text}"
                )

        transcript = " ".join(
            transcript_parts
        ).strip()

        if not transcript:
            transcript = "[NO TRANSCRIPTION]"

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