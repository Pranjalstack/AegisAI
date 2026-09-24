# backend/live_audio.py

import time
import wave
from pathlib import Path

import sounddevice as sd


CAPTURE_SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2

OUTPUT_DIR = Path("audio")


def get_default_input_device():
    """
    Use the operating system's default input device.
    """

    device = sd.query_devices(
        kind="input"
    )

    if device["max_input_channels"] < 1:
        raise RuntimeError(
            "The default input device has no microphone channels."
        )

    return device


def record_chunk(
    output_path,
    duration=5,
):
    """
    Record microphone audio at the device's native 44.1 kHz
    sample rate and save it as mono PCM WAV.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    duration = float(duration)

    if duration <= 0:
        raise ValueError(
            "Recording duration must be greater than 0."
        )

    if duration > 30:
        raise ValueError(
            "Recording duration cannot exceed 30 seconds."
        )

    device = get_default_input_device()

    device_name = device["name"]

    native_rate = float(
        device["default_samplerate"]
    )

    print(
        "\n===== AEGIS LIVE AUDIO CAPTURE =====",
        flush=True,
    )

    print(
        f"Input device: {device_name}",
        flush=True,
    )

    print(
        f"Device sample rate: {native_rate:.0f} Hz",
        flush=True,
    )

    print(
        f"Capture sample rate: {CAPTURE_SAMPLE_RATE} Hz",
        flush=True,
    )

    print(
        f"Duration: {duration:.1f} seconds",
        flush=True,
    )

    print(
        "Recording...",
        flush=True,
    )

    frames = int(
        CAPTURE_SAMPLE_RATE * duration
    )

    recording = sd.rec(
        frames,
        samplerate=CAPTURE_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=None,
    )

    sd.wait()

    print(
        "Recording complete.",
        flush=True,
    )

    with wave.open(
        str(output_path),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            CHANNELS
        )

        wav_file.setsampwidth(
            SAMPLE_WIDTH_BYTES
        )

        wav_file.setframerate(
            CAPTURE_SAMPLE_RATE
        )

        wav_file.writeframes(
            recording.tobytes()
        )

    print(
        f"Saved: {output_path}",
        flush=True,
    )

    return str(
        output_path
    )


def record_test_chunk():
    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        OUTPUT_DIR
        / f"live_chunk_{timestamp}.wav"
    )

    return record_chunk(
        output_path,
        duration=5,
    )


if __name__ == "__main__":

    path = record_test_chunk()

    print(
        "\n===== LIVE AUDIO TEST COMPLETE ====="
    )

    print(
        f"Audio file: {path}"
    )