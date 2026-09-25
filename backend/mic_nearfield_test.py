import time
import wave
from pathlib import Path

import sounddevice as sd


OUTPUT_DIR = Path("audio")
DEVICE_ID = 1
SAMPLE_RATE = 44100
CHANNELS = 2
DURATION = 5


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

    info = sd.query_devices(
        DEVICE_ID,
        "input",
    )

    print("\nDevice name:")
    print(info["name"])

    print("\nIMPORTANT:")
    print("1. Go somewhere quiet.")
    print("2. Keep the laptop microphone about 10-20 cm from your mouth.")
    print("3. Make sure nobody else is speaking nearby.")
    print("4. Do not play any other audio.")
    print("5. Say this sentence clearly:")

    print("\nThis is an Aegis microphone test.")

    print("\nRecording in 3 seconds...")
    time.sleep(3)

    print("Recording...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=DEVICE_ID,
    )

    sd.wait()

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        OUTPUT_DIR
        / f"nearfield_{timestamp}.wav"
    )

    save_wav(path, audio)

    print("\nRecording complete.")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()