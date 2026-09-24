# backend/live_audio_scan.py

from pathlib import Path
import subprocess

from backend.audio_analyzer import analyze_audio_with_ai


def enhance_audio(
    input_path,
    output_path,
):
    """
    Prepare quiet microphone audio for Whisper.

    The microphone diagnostic showed a low average level,
    so we apply moderate gain and basic speech-frequency
    filtering before transcription.
    """

    input_path = Path(
        input_path
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),

        "-af",
        (
            "highpass=f=80,"
            "lowpass=f=8000,"
            "volume=12dB"
        ),

        "-ar",
        "16000",

        "-ac",
        "1",

        "-c:a",
        "pcm_s16le",

        str(output_path),
    ]

    subprocess.run(
        command,
        check=True,
    )

    return str(
        output_path
    )


def find_latest_chunk():
    """
    Find the newest live microphone chunk.
    """

    audio_dir = Path(
        "audio"
    )

    chunks = list(
        audio_dir.glob(
            "live_chunk_*.wav"
        )
    )

    if not chunks:
        raise FileNotFoundError(
            "No live microphone chunks were found."
        )

    return max(
        chunks,
        key=lambda path: path.stat().st_mtime,
    )


def analyze_live_chunk(
    audio_path,
):
    """
    Enhance the live microphone chunk and run
    Whisper + threat detection + Qwen.
    """

    input_path = Path(
        audio_path
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Audio chunk not found: {audio_path}"
        )

    enhanced_path = (
        Path("audio")
        / "live_enhanced.wav"
    )

    print(
        "\n===== AEGIS LIVE AUDIO ANALYSIS =====",
        flush=True,
    )

    print(
        f"Input: {input_path}",
        flush=True,
    )

    print(
        "[1/2] Enhancing microphone audio...",
        flush=True,
    )

    enhance_audio(
        input_path,
        enhanced_path,
    )

    print(
        "[1/2] Audio enhancement complete.",
        flush=True,
    )

    print(
        "[2/2] Running local Whisper + threat analysis + AI...",
        flush=True,
    )

    result = analyze_audio_with_ai(
        str(enhanced_path)
    )

    scan = result["scan"]

    print(
        "\n===== LIVE AUDIO RESULT ====="
    )

    print(
        f"Risk Level: {scan['risk']}"
    )

    print(
        f"Risk Score: {scan['score']}"
    )

    print(
        "\nTranscript:"
    )

    if scan["text"]:
        print(
            scan["text"]
        )
    else:
        print(
            "[No speech detected]"
        )

    print(
        "\nDetected indicators:"
    )

    if scan["reasons"]:

        for reason in scan["reasons"]:
            print(
                f"• {reason}"
            )

    else:

        print(
            "• None"
        )

    if scan["urls"]:

        print(
            "\nLinks detected:"
        )

        for url in scan["urls"]:
            print(
                f"• {url}"
            )

    print(
        "\n===== AI EXPLANATION ====="
    )

    print(
        result["explanation"]
    )


if __name__ == "__main__":

    latest_chunk = find_latest_chunk()

    analyze_live_chunk(
        latest_chunk
    )