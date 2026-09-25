from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


AUDIO_DIR = Path("audio")
MODEL_NAME = "openai/whisper-base"


def latest_nearfield_recording():
    files = sorted(
        AUDIO_DIR.glob("nearfield_*.wav")
    )

    return files[-1] if files else None


def load_whisper():
    print("===== LOADING LOCAL WHISPER =====")

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32

    if use_cuda:
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )
    else:
        print("Using CPU")

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
    print("===== AEGIS NEAR-FIELD WHISPER TEST =====")

    audio_path = latest_nearfield_recording()

    if audio_path is None:
        print("\nNo near-field recording found.")
        print("Expected: audio\\nearfield_*.wav")
        return

    print(f"\nTesting:")
    print(audio_path)

    whisper = load_whisper()

    print("\n===== TRANSCRIPTION =====")

    try:
        transcript = transcribe(
            whisper,
            audio_path,
        )

        if not transcript:
            transcript = "[NO TRANSCRIPTION]"

        print(f"\nTranscript:\n{transcript}")

    except Exception as exc:
        print(
            f"\nWhisper failed: "
            f"{type(exc).__name__}: {exc}"
        )

    print("\n===== TEST COMPLETE =====")
    print("\nExpected:")
    print("This is an Aegis microphone test.")


if __name__ == "__main__":
    main()