from pathlib import Path
import wave

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


MODEL_NAME = "openai/whisper-base"
AUDIO_DIR = Path("audio")


def latest_file(pattern):
    files = sorted(AUDIO_DIR.glob(pattern))
    return files[-1] if files else None


def audio_info(path):
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()

    duration = frames / sample_rate if sample_rate else 0

    return {
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
    }


def load_whisper():
    print("\n===== LOADING WHISPER =====")

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32

    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU mode")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

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


def transcribe_with_guards(whisper, path):
    result = whisper(
        str(path),
        return_timestamps=True,
        generate_kwargs={
            "task": "transcribe",
            "language": "en",
            "temperature": 0,
            "num_beams": 5,

            # Reduce repetitive/hallucinated decoding
            "condition_on_prev_tokens": False,

            # Whisper confidence / hallucination guards
            "compression_ratio_threshold": 1.35,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        },
    )

    text = result.get("text", "").strip()
    chunks = result.get("chunks", [])

    return text, chunks


def print_result(label, path, whisper):
    print(f"\n===== {label} =====")
    print(f"File: {path}")

    info = audio_info(path)

    print(f"Duration   : {info['duration']:.2f} sec")
    print(f"Sample rate: {info['sample_rate']} Hz")
    print(f"Channels   : {info['channels']}")

    try:
        text, chunks = transcribe_with_guards(
            whisper,
            path,
        )

        print(f"\nTranscript:\n{text}")

        if chunks:
            print("\nTimestamp chunks:")

            for chunk in chunks[:20]:
                timestamp = chunk.get("timestamp")
                chunk_text = chunk.get("text", "").strip()

                print(
                    f"{timestamp} -> {chunk_text}"
                )

            if len(chunks) > 20:
                print(
                    f"... {len(chunks) - 20} more chunks"
                )

        # Simple hallucination indicator
        words = text.split()

        if len(words) > 35 and info["duration"] < 6:
            print(
                "\n⚠️ WARNING: Transcript is unusually long "
                "for this recording duration."
            )

        if len(words) >= 10:
            half = len(words) // 2
            first_half = " ".join(words[:half])
            second_half = " ".join(words[half:])

            if first_half == second_half:
                print(
                    "⚠️ WARNING: Strong repeated-text pattern detected."
                )

    except Exception as exc:
        print(
            f"Transcription failed: "
            f"{type(exc).__name__}: {exc}"
        )


def main():
    print("===== AEGIS WHISPER HALLUCINATION GUARD TEST =====")

    device_1 = latest_file("mic_device_1_*.wav")
    test_call = AUDIO_DIR / "test_call.wav"

    files = []

    if device_1:
        files.append(
            ("MICROPHONE DEVICE 1", device_1)
        )
    else:
        print("Device 1 recording not found.")

    if test_call.exists():
        files.append(
            ("KNOWN AEGIS TEST CALL", test_call)
        )
    else:
        print("audio\\test_call.wav not found.")

    if not files:
        print("No test files available.")
        return

    whisper = load_whisper()

    for label, path in files:
        print_result(
            label,
            path,
            whisper,
        )

    print("\n===== GUARD TEST COMPLETE =====")


if __name__ == "__main__":
    main()