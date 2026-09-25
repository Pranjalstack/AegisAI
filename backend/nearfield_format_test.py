from pathlib import Path
import subprocess

from faster_whisper import WhisperModel


AUDIO_DIR = Path("audio")
MODEL_SIZE = "base"


def latest_nearfield():
    files = sorted(
        AUDIO_DIR.glob("nearfield_*.wav")
    )

    return files[-1] if files else None


def convert_audio(input_path):
    output_path = input_path.with_name(
        f"{input_path.stem}_16k_mono.wav"
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg conversion failed:\n"
            + result.stderr
        )

    return output_path


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
            print(
                f"[{segment.start:.2f}s -> "
                f"{segment.end:.2f}s] {text}"
            )
            parts.append(text)

    return " ".join(parts).strip(), info


def main():
    print("===== AEGIS MICROPHONE FORMAT TEST =====")

    original = latest_nearfield()

    if original is None:
        print("\nNo near-field recording found.")
        return

    print(f"\nOriginal:")
    print(original)

    print("\nConverting:")
    print("44.1 kHz stereo → 16 kHz mono")

    try:
        converted = convert_audio(original)
    except Exception as exc:
        print(
            f"\nConversion failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    print(f"Converted:")
    print(converted)

    print("\n===== LOADING FASTER-WHISPER =====")

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

    print("\n===== TRANSCRIPTION =====")

    try:
        text, info = transcribe(
            model,
            converted,
        )

        if not text:
            text = "[NO TRANSCRIPTION]"

        print(
            f"\nDetected language: "
            f"{info.language}"
        )

        print(
            f"Language probability: "
            f"{info.language_probability:.3f}"
        )

        print(f"\nFinal transcript:")
        print(text)

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