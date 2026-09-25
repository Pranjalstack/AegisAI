# backend/live_audio.py

import time
import wave
from pathlib import Path

import sounddevice as sd

from backend.audio_analyzer import analyze_audio_with_ai


OUTPUT_DIR = Path("audio")

DEVICE_ID = 1
SAMPLE_RATE = 44100
CHANNELS = 1
DURATION = 8


def record_chunk():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        OUTPUT_DIR
        / f"live_chunk_{timestamp}.wav"
    )

    print(
        "===== AEGIS LIVE MICROPHONE TEST ====="
    )

    try:
        info = sd.query_devices(
            DEVICE_ID,
            "input",
        )

        print(
            f"Microphone: {info['name']}"
        )

        print(
            f"Sample rate: {SAMPLE_RATE} Hz"
        )

        print(
            f"Channels: {CHANNELS}"
        )

        print(
            f"Duration: {DURATION} seconds"
        )

    except Exception as exc:
        raise RuntimeError(
            "Unable to query microphone: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    print("\nYou will be recorded for 8 seconds.")
    print("Speak naturally and clearly.")
    print(
        "\nExample:"
        "\nYour bank account will be suspended today."
        " Please give me your OTP to verify your account."
    )

    print("\nRecording starts in 3 seconds...")
    time.sleep(3)

    print("Recording...")

    try:
        audio = sd.rec(
            int(
                DURATION * SAMPLE_RATE
            ),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            device=DEVICE_ID,
        )

        sd.wait()

    except Exception as exc:
        raise RuntimeError(
            "Microphone recording failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        with wave.open(
            str(output_path),
            "wb",
        ) as wf:
            wf.setnchannels(
                CHANNELS
            )
            wf.setsampwidth(2)
            wf.setframerate(
                SAMPLE_RATE
            )
            wf.writeframes(
                audio.tobytes()
            )

    except Exception as exc:
        raise RuntimeError(
            "Could not save WAV recording: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    print("\nRecording complete.")
    print(
        f"Saved: {output_path}"
    )

    return output_path


def analyze_chunk(audio_path):
    print(
        "\n===== AEGIS AUDIO ANALYSIS ====="
    )

    print(
        f"Input: {audio_path}"
    )

    print(
        "\nRunning local Faster-Whisper "
        "small.en..."
    )

    try:
        result = analyze_audio_with_ai(
            str(audio_path)
        )

    except Exception as exc:
        print(
            "\nAudio analysis failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None

    scan = result.get(
        "scan",
        {},
    )

    metadata = scan.get(
        "metadata",
        {},
    )

    print(
        "\n===== LIVE AUDIO RESULT ====="
    )

    print(
        f"Risk Level: "
        f"{scan.get('risk', 'LOW')}"
    )

    print(
        f"Risk Score: "
        f"{scan.get('score', 0)}"
    )

    print(
        "\nTranscript:"
    )

    transcript = scan.get(
        "text",
        "",
    )

    if transcript:
        print(transcript)
    else:
        print(
            "[No reliable transcription]"
        )

    print(
        "\nTranscription status:"
    )

    print(
        metadata.get(
            "analysis_status",
            "UNKNOWN",
        )
    )

    if "confidence_percent" in metadata:
        print(
            "Confidence: "
            f"{metadata['confidence_percent']:.0f}%"
        )

    print(
        "\nDetected indicators:"
    )

    reasons = scan.get(
        "reasons",
        [],
    )

    if reasons:
        for reason in reasons:
            print(
                f"• {reason}"
            )
    else:
        print(
            "• None"
        )

    urls = scan.get(
        "urls",
        [],
    )

    if urls:
        print(
            "\nLinks detected:"
        )

        for url in urls:
            print(
                f"• {url}"
            )

    print(
        "\n===== AI EXPLANATION ====="
    )

    print(
        result.get(
            "explanation",
            "",
        )
    )

    return result


def main():
    try:
        audio_path = record_chunk()

        analyze_chunk(
            audio_path
        )

    except KeyboardInterrupt:
        print(
            "\nRecording cancelled."
        )

    except Exception as exc:
        print(
            f"\nAegis live audio failed: "
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()