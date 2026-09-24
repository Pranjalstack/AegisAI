from pathlib import Path
import subprocess
import wave

import numpy as np


AUDIO_DIR = Path("audio")


def latest_raw_recording():
    files = sorted(AUDIO_DIR.glob("live_chunk_*.wav"))

    files = [
        file
        for file in files
        if "_clean" not in file.stem
        and "_enhanced" not in file.stem
    ]

    return files[-1] if files else None


def analyze_wave(path):
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM, got sample width {sample_width}"
        )

    audio = (
        np.frombuffer(raw, dtype=np.int16)
        .astype(np.float32)
        / 32768.0
    )

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    duration = len(audio) / sample_rate

    rms = float(np.sqrt(np.mean(audio ** 2)))

    peak = float(np.max(np.abs(audio)))

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

    # Samples extremely close to digital clipping.
    clipping_threshold = 0.995
    clipped_samples = np.sum(
        np.abs(audio) >= clipping_threshold
    )

    clipping_percent = (
        clipped_samples / len(audio) * 100
        if len(audio)
        else 0
    )

    # Approximate active-speech/noise activity.
    # This is NOT a full VAD. It simply estimates how much
    # of the recording is above several amplitude thresholds.
    thresholds_db = [-40, -35, -30, -25, -20]

    activity = {}

    for threshold_db in thresholds_db:
        threshold = 10 ** (threshold_db / 20)

        active = np.sum(
            np.abs(audio) >= threshold
        )

        activity[threshold_db] = (
            active / len(audio) * 100
            if len(audio)
            else 0
        )

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "duration": duration,
        "rms_db": rms_db,
        "peak_db": peak_db,
        "clipping_percent": clipping_percent,
        "activity": activity,
    }


def ffmpeg_silence_analysis(path):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(path),
        "-af",
        "silencedetect=n=-35dB:d=0.20",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    output = result.stderr

    silence_lines = [
        line.strip()
        for line in output.splitlines()
        if "silence_" in line
    ]

    return silence_lines


def main():
    print("===== AEGIS AUDIO QUALITY TEST =====")

    path = latest_raw_recording()

    if path is None:
        print("No raw live recording found.")
        print(
            "Expected: audio\\live_chunk_*.wav"
        )
        return

    print(f"\nFile: {path}")

    try:
        info = analyze_wave(path)
    except Exception as exc:
        print(
            f"Audio analysis failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    print("\n===== WAV PROPERTIES =====")
    print(
        f"Sample rate       : {info['sample_rate']} Hz"
    )
    print(
        f"Channels          : {info['channels']}"
    )
    print(
        f"Duration          : {info['duration']:.2f} sec"
    )
    print(
        f"RMS               : {info['rms_db']:.2f} dBFS"
    )
    print(
        f"Peak              : {info['peak_db']:.2f} dBFS"
    )
    print(
        f"Near-clipped      : "
        f"{info['clipping_percent']:.4f}%"
    )

    print("\n===== LEVEL ACTIVITY =====")

    for threshold_db, percent in info["activity"].items():
        print(
            f"Above {threshold_db:>3} dBFS:"
            f" {percent:6.2f}%"
        )

    print("\n===== FFMPEG SILENCE DETECTION =====")

    try:
        silence = ffmpeg_silence_analysis(path)

        if silence:
            for line in silence:
                print(line)
        else:
            print(
                "No silence segments detected "
                "at the selected threshold."
            )

    except FileNotFoundError:
        print("FFmpeg was not found in PATH.")

    print("\n===== INTERPRETATION GUIDE =====")

    if info["clipping_percent"] > 0.1:
        print(
            "⚠️ Significant clipping detected. "
            "Lower the Windows microphone input level."
        )
    elif info["clipping_percent"] > 0:
        print(
            "⚠️ Some samples are near clipping. "
            "Input level may be slightly high."
        )
    else:
        print(
            "✅ No meaningful digital clipping detected."
        )

    if info["rms_db"] < -35:
        print(
            "⚠️ Average level is relatively quiet."
        )
    elif info["rms_db"] > -15:
        print(
            "⚠️ Average level is very high."
        )
    else:
        print(
            "✅ Average recording level is reasonable."
        )

    print("\n===== TEST COMPLETE =====")


if __name__ == "__main__":
    main()