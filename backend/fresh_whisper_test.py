from pathlib import Path
import wave

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


AUDIO_DIR = Path("audio")
MODEL_NAME = "openai/whisper-base"


def latest_raw_recording():
    files = sorted(AUDIO_DIR.glob("live_chunk_*.wav"))

    # Safety:
    files = [
        file for file in files
        if "_enhanced" not in file.stem
        and "_clean" not in file.stem
    ]

    return files[-1] if files else None


def inspect_audio(path):
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width == 2:
        audio = (
            np.frombuffer(raw, dtype=np.int16)
            .astype(np.float32)
            / 32768.0
        )
    else:
        raise ValueError(
            f"Unsupported sample width: {sample_width}"
        )

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    rms_db = -120.0 if rms <= 1e-9 else 20 * np.log10(rms)
    peak_db = -120.0 if peak <= 1e-9 else 20 * np.log10(peak)

    duration = frames / sample_rate

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": duration,
        "rms_db": rms_db,
        "peak_db": peak_db,
    }


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
    print("===== AEGIS FRESH RAW AUDIO TEST =====")

    path = latest_raw_recording()

    if path is None:
        print(
            "\nNo fresh raw recording found."
        )
        print(
            "Expected: audio\\live_chunk_*.wav"
        )
        return

    print(f"\nRecording:")
    print(path)

    info = inspect_audio(path)

    print("\n===== AUDIO INFORMATION =====")
    print(
        f"Sample rate : {info['sample_rate']} Hz"
    )
    print(
        f"Channels    : {info['channels']}"
    )
    print(
        f"Duration    : {info['duration']:.2f} sec"
    )
    print(
        f"RMS         : {info['rms_db']:.2f} dBFS"
    )
    print(
        f"Peak        : {info['peak_db']:.2f} dBFS"
    )

    whisper = load_whisper()

    print("\n===== WHISPER RESULT =====")

    try:
        transcript = transcribe(
            whisper,
            path,
        )

        if not transcript:
            transcript = "[NO TRANSCRIPTION]"

        print(f"\nTranscript:\n{transcript}")

    except Exception as exc:
        print(
            f"Whisper failed: "
            f"{type(exc).__name__}: {exc}"
        )

    print("\n===== FRESH TEST COMPLETE =====")
    print(
        "\nExpected speech:"
    )
    print(
        "This is an Aegis microphone test."
    )


if __name__ == "__main__":
    main()