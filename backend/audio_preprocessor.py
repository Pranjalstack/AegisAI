import subprocess
from pathlib import Path


AUDIO_DIR = Path("audio")


def enhance_audio(input_path, output_path=None):
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input audio not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_clean.wav"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_chain = (
        "highpass=f=80,"
        "lowpass=f=8000,"
        "afftdn=nr=12:nf=-25:tn=1,"
        "dynaudnorm=f=150:g=15"
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(input_path),
        "-af",
        filter_chain,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg audio enhancement failed:\n"
            f"{result.stderr}"
        )

    return output_path


def latest_recording(device_id):
    files = sorted(
        AUDIO_DIR.glob(f"mic_device_{device_id}_*.wav")
    )

    if not files:
        return None

    return files[-1]


def main():
    print("===== AEGIS OFFLINE AUDIO ENHANCEMENT =====")

    for device_id in (1, 11):
        input_file = latest_recording(device_id)

        if input_file is None:
            print(f"\nDevice {device_id}: no recording found.")
            continue

        output_file = input_file.with_name(
            f"{input_file.stem}_clean.wav"
        )

        try:
            print(f"\nDevice {device_id}")
            print(f"Input : {input_file}")
            print(f"Output: {output_file}")

            enhance_audio(input_file, output_file)

            print("Enhancement complete.")

        except Exception as exc:
            print(
                f"Device {device_id} failed: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n===== AUDIO ENHANCEMENT COMPLETE =====")


if __name__ == "__main__":
    main()