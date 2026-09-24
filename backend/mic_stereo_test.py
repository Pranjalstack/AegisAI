import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


OUTPUT_DIR = Path("audio")
DEVICE_ID = 1
SAMPLE_RATE = 44100
CHANNELS = 2
DURATION = 4


def save_wav(path, audio, sample_rate):
    audio = np.asarray(audio, dtype=np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(audio.shape[1] if audio.ndim > 1 else 1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== AEGIS MICROPHONE ARRAY STEREO TEST =====")
    print(f"Device: {DEVICE_ID}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS}")
    print(f"Duration: {DURATION} seconds")

    info = sd.query_devices(
        DEVICE_ID,
        "input",
    )

    print(f"\nDevice name: {info['name']}")
    print(
        f"Available input channels: "
        f"{info['max_input_channels']}"
    )

    if info["max_input_channels"] < 2:
        print(
            "\nThis device does not provide "
            "two input channels."
        )
        return

    timestamp = time.strftime(
        "%Y%m%d_%H%M%S"
    )

    stereo_path = (
        OUTPUT_DIR
        / f"mic_stereo_{timestamp}.wav"
    )

    left_path = (
        OUTPUT_DIR
        / f"mic_left_{timestamp}.wav"
    )

    right_path = (
        OUTPUT_DIR
        / f"mic_right_{timestamp}.wav"
    )

    mono_path = (
        OUTPUT_DIR
        / f"mic_mono_{timestamp}.wav"
    )

    print("\nSay clearly:")
    print("This is an Aegis microphone test.")

    print("\nRecording in 2 seconds...")
    time.sleep(2)

    print("Recording...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        device=DEVICE_ID,
    )

    sd.wait()

    print("Recording complete.")

    save_wav(
        stereo_path,
        audio,
        SAMPLE_RATE,
    )

    left = audio[:, 0]
    right = audio[:, 1]

    save_wav(
        left_path,
        left,
        SAMPLE_RATE,
    )

    save_wav(
        right_path,
        right,
        SAMPLE_RATE,
    )

    mono = (
        (
            left.astype(np.int32)
            + right.astype(np.int32)
        )
        // 2
    ).astype(np.int16)

    save_wav(
        mono_path,
        mono,
        SAMPLE_RATE,
    )

    print("\n===== FILES CREATED =====")
    print(f"Stereo : {stereo_path}")
    print(f"Left   : {left_path}")
    print(f"Right  : {right_path}")
    print(f"Mono   : {mono_path}")

    print("\n===== CHANNEL LEVELS =====")

    for name, channel in [
        ("Left", left),
        ("Right", right),
        ("Mono", mono),
    ]:
        values = (
            channel.astype(np.float32)
            / 32768.0
        )

        rms = float(
            np.sqrt(np.mean(values ** 2))
        )

        peak = float(
            np.max(np.abs(values))
        )

        rms_db = (
            -120.0
            if rms <= 1e-12
            else 20 * np.log10(rms)
        )

        peak_db = (
            -120.0
            if peak <= 1e-12
            else 20 * np.log10(peak)
        )

        print(
            f"{name:5} | "
            f"RMS {rms_db:7.2f} dBFS | "
            f"Peak {peak_db:7.2f} dBFS"
        )

    print("\n===== TEST COMPLETE =====")


if __name__ == "__main__":
    main()