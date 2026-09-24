# backend/mic_compare.py

import time
import wave
from pathlib import Path

import sounddevice as sd


SAMPLE_RATE = 44100
CHANNELS = 1
DURATION = 3

TEST_DEVICES = [
    1,
    9,
    11,
    17,
]


def record_device(device_id, output_path):

    info = sd.query_devices(
        device_id,
        "input"
    )

    name = info["name"]

    print(
        f"\nDevice {device_id}: {name}"
    )

    print(
        "Recording in 2 seconds..."
    )

    time.sleep(2)

    frames = int(
        SAMPLE_RATE * DURATION
    )

    print(
        "Recording..."
    )

    recording = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=device_id,
    )

    sd.wait()

    print(
        "Recording complete."
    )

    with wave.open(
        str(output_path),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            CHANNELS
        )

        wav_file.setsampwidth(
            2
        )

        wav_file.setframerate(
            SAMPLE_RATE
        )

        wav_file.writeframes(
            recording.tobytes()
        )

    print(
        f"Saved: {output_path}"
    )


def main():

    output_dir = Path(
        "audio"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n===== AEGIS MICROPHONE COMPARISON ====="
    )

    print(
        "You will record a separate 3-second sample for each available device."
    )

    print(
        "Say the same sentence every time:"
    )

    print(
        "This is an Aegis microphone test."
    )

    for device_id in TEST_DEVICES:

        try:

            info = sd.query_devices(
                device_id,
                "input"
            )

            if info["max_input_channels"] < 1:
                continue

            timestamp = time.strftime(
                "%Y%m%d_%H%M%S"
            )

            output_path = (
                output_dir
                / f"mic_device_{device_id}_{timestamp}.wav"
            )

            record_device(
                device_id,
                output_path,
            )

        except Exception as error:

            print(
                f"Device {device_id} failed: "
                f"{type(error).__name__}: {error}"
            )

    print(
        "\n===== MICROPHONE COMPARISON COMPLETE ====="
    )


if __name__ == "__main__":
    main()