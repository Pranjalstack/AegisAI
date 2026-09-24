from pathlib import Path
import subprocess

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


AUDIO_DIR = Path("audio")
REFERENCE_FILE = AUDIO_DIR / "aegis_reference.wav"
MODEL_NAME = "openai/whisper-base"

EXPECTED_TEXT = "This is an Aegis microphone test."


def generate_reference_audio():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("===== GENERATING OFFLINE REFERENCE AUDIO =====")

    powershell_script = f"""
Add-Type -AssemblyName System.Speech

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

$synth.SetOutputToWaveFile("{REFERENCE_FILE.resolve()}")

$synth.Speak("{EXPECTED_TEXT}")

$synth.Dispose()
"""

    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            powershell_script,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Windows offline speech generation failed:\n"
            + result.stderr
        )

    if not REFERENCE_FILE.exists():
        raise RuntimeError(
            "Reference WAV was not created."
        )

    print(f"Reference created:")
    print(REFERENCE_FILE)


def convert_reference_audio():
    converted = AUDIO_DIR / "aegis_reference_16k.wav"

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(REFERENCE_FILE),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(converted),
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

    return converted


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
        print("Using CPU.")

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


def transcribe(whisper, audio_path):
    result = whisper(
        str(audio_path),
        generate_kwargs={
            "task": "transcribe",
            "language": "en",
            "temperature": 0,
            "condition_on_prev_tokens": False,
        },
    )

    return result.get("text", "").strip()


def main():
    print("===== AEGIS WHISPER REFERENCE TEST =====")
    print(f'Expected: "{EXPECTED_TEXT}"')

    try:
        generate_reference_audio()

        reference_16k = convert_reference_audio()

        print(
            f"\nConverted reference:"
            f"\n{reference_16k}"
        )

        whisper = load_whisper()

        print("\n===== WHISPER RESULT =====")

        transcript = transcribe(
            whisper,
            reference_16k,
        )

        if not transcript:
            transcript = "[NO TRANSCRIPTION]"

        print(f"\nTranscript:\n{transcript}")

        normalized_expected = (
            EXPECTED_TEXT.lower()
            .replace(".", "")
        )

        normalized_result = (
            transcript.lower()
            .replace(".", "")
        )

        if normalized_expected in normalized_result:
            print(
                "\n✅ REFERENCE TEST PASSED"
            )
            print(
                "Whisper can correctly transcribe "
                "known speech."
            )
        else:
            print(
                "\n⚠️ REFERENCE TEST DID NOT MATCH"
            )
            print(
                "The Whisper pipeline itself now "
                "needs investigation."
            )

    except Exception as exc:
        print(
            f"\nTest failed: "
            f"{type(exc).__name__}: {exc}"
        )

    print("\n===== REFERENCE TEST COMPLETE =====")


if __name__ == "__main__":
    main()