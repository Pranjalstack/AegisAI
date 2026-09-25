# backend/live_audio_scan.py

from pathlib import Path

from backend.audio_analyzer import (
    analyze_audio_with_ai,
)


AUDIO_DIR = Path("audio")


def find_latest_raw_chunk():
    chunks = sorted(
        AUDIO_DIR.glob("live_chunk_*.wav")
    )

    raw_chunks = [
        path
        for path in chunks
        if "_clean" not in path.stem
        and "_enhanced" not in path.stem
        and "_trimmed" not in path.stem
    ]

    if not raw_chunks:
        raise FileNotFoundError(
            "No raw live microphone chunks found."
        )

    return raw_chunks[-1]


def main():
    print(
        "===== AEGIS LIVE AUDIO ANALYSIS ====="
    )

    audio_path = find_latest_raw_chunk()

    print(
        f"\nUsing raw recording:"
        f"\n{audio_path}"
    )

    print(
        "\nRunning local Faster-Whisper "
        "small.en..."
    )

    result = analyze_audio_with_ai(
        str(audio_path)
    )

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

    text = scan.get(
        "text",
        "",
    )

    print(
        text
        if text
        else "[No reliable transcription]"
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


if __name__ == "__main__":
    main()