# backend/mic_diagnostics.py

import math
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 5

OUTPUT_PATH = (
    Path("audio")
    / "mic_diagnostic.wav"
)


def dbfs(value):
    if value <= 0:
        return float("-inf")

    return 20 * math.log10(
        value
    )


def main():

    print(
        "\n===== AEGIS MICROPHONE DIAGNOSTIC ====="
    )

    default_input = sd.query_devices(
        kind="input"
    )

    print(
        "\nDefault input device:"
    )

    print(
        f"Name: {default_input['name']}"
    )

    print(
        f"Channels: {default_input['max_input_channels']}"
    )

    print(
        f"Sample rate: {default_input['default_samplerate']}"
    )

    if default_input["max_input_channels"] < 1:

        raise RuntimeError(
            "No microphone input device is available."
        )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    print(
        "\nSpeak clearly for the entire 5 seconds:"
    )

    print(
        "This is an Aegis microphone diagnostic test."
    )

    print(
        "\nRecording..."
    )


    recording = sd.rec(
        int(
            SAMPLE_RATE * DURATION
        ),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=None,
    )

    sd.wait()


    print(
        "Recording complete."
    )


    audio = recording.astype(
        np.float32
    ) / 32768.0

    audio = audio.reshape(
        -1
    )


    rms = float(
        np.sqrt(
            np.mean(
                audio ** 2
            )
        )
    )

    peak = float(
        np.max(
            np.abs(audio)
        )
    )

    non_silent = float(
        np.mean(
            np.abs(audio) > 0.01
        )
    )


    with wave.open(
        str(OUTPUT_PATH),
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
        "\n===== MICROPHONE ANALYSIS ====="
    )

    print(
        f"RMS: {rms:.6f}"
    )

    print(
        f"RMS dBFS: {dbfs(rms):.2f} dB"
    )

    print(
        f"Peak: {peak:.6f}"
    )

    print(
        f"Peak dBFS: {dbfs(peak):.2f} dB"
    )

    print(
        f"Samples above 1% amplitude: "
        f"{non_silent * 100:.2f}%"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()