# backend/mic_nearfield_test.py

import time
import wave
from pathlib import Path

import sounddevice as sd


OUTPUT_DIR = Path("audio")
DEVICE_ID = 1
SAMPLE_RATE = 44100
CHANNELS = 2
DURATION = 8


def save_wav(path, audio):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== AEGIS NEAR-FIELD MICROPHONE TEST =====")
    print(f"Device      : {DEVICE_ID}")
    print(f"Sample rate : {SAMPLE_RATE} Hz")
    print(f"Channels    : {CHANNELS}")
    print(f"Duration    : {DURATION} sec")

    try:
        info = sd.query_devices(
            DEVICE_ID,
            "input",
        )
    except Exception as exc:
        print(
            f"\nUnable to query microphone: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    print("\nDevice name:")
    print(info["name"])

    print("\nIMPORTANT:")
    print("1. Go somewhere as quiet as possible.")
    print("2. Keep the laptop microphone about 10-20 cm from your mouth.")
    print("3. Make sure nobody else is speaking nearby.")
    print("4. Do not play any other audio.")
    print("5. Speak at a normal, clear volume.")
    print("\nSay exactly:")

    print(
        "\nYour bank account will be suspended today. "
        "Please give me your OTP to verify your account."
    )

    print("\nRecording in 3 seconds...")
    time.sleep(3)

    print("Recording...")

    try:
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            device=DEVICE_ID,
        )

        sd.wait()

    except Exception as exc:
        print(
            f"\nRecording failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        OUTPUT_DIR
        / f"nearfield_{timestamp}.wav"
    )

    try:
        save_wav(path, audio)
    except Exception as exc:
        print(
            f"\nFailed to save recording: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    print("\nRecording complete.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()