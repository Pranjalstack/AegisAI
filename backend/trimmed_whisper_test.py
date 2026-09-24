from pathlib import Path
import subprocess

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


AUDIO_DIR = Path("audio")
MODEL_NAME = "openai/whisper-base"


def latest_raw_recording():
    files = sorted(AUDIO_DIR.glob("live_chunk_*.wav"))

    files = [
        file
        for file in files
        if "_clean" not in file.stem
        and "_enhanced" not in file.stem
        and "_trimmed" not in file.stem
    ]

    return files[-1] if files else None


def trim_silence(input_path, output_path):
    filter_chain = (
        "silenceremove="
        "start_periods=1:"
        "start_duration=0.20:"
        "start_threshold=-35dB:"
        "stop_periods=1:"
        "stop_duration=0.20:"
        "stop_threshold=-35dB"
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(input_path),
        "-af",
        filter_chain,
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
            "FFmpeg silence removal failed:\n"
            + result.stderr
        )


def get_duration(path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def load_whisper():
    print("\n===== LOADING LOCAL WHISPER =====")

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32

    if use_cuda:
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print("GPU unavailable. Using CPU.")

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

    return result.get("text", "").strip()


def main():
    print("===== AEGIS SILENCE-TRIMMED WHISPER TEST =====")

    original = latest_raw_recording()

    if original is None:
        print("No raw microphone recording found.")
        return

    trimmed = original.with_name(
        f"{original.stem}_trimmed.wav"
    )

    print(f"\nOriginal:")
    print(original)

    print("\nRemoving leading/trailing silence...")

    try:
        trim_silence(
            original,
            trimmed,
        )
    except Exception as exc:
        print(
            f"Trimming failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    original_duration = get_duration(original)
    trimmed_duration = get_duration(trimmed)

    print("\n===== DURATIONS =====")

    if original_duration is not None:
        print(
            f"Original : "
            f"{original_duration:.2f} sec"
        )
    else:
        print("Original : unknown")

    if trimmed_duration is not None:
        print(
            f"Trimmed  : "
            f"{trimmed_duration:.2f} sec"
        )
    else:
        print("Trimmed  : unknown")

    print(f"\nTrimmed file:")
    print(trimmed)

    whisper = load_whisper()

    print("\n===== TRIMMED WHISPER RESULT =====")

    try:
        transcript = transcribe(
            whisper,
            trimmed,
        )

        if not transcript:
            transcript = "[NO TRANSCRIPTION]"

        print(f"\nTranscript:\n{transcript}")

    except Exception as exc:
        print(
            f"Whisper failed: "
            f"{type(exc).__name__}: {exc}"
        )

    print("\n===== TEST COMPLETE =====")

    print("\nExpected speech:")
    print("This is an Aegis microphone test.")


if __name__ == "__main__":
    main()