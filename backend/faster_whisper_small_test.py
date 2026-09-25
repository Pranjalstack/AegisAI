# backend/faster_whisper_small_test.py

from pathlib import Path

from faster_whisper import WhisperModel


AUDIO_DIR = Path("audio")
MODEL_SIZE = "small.en"

# Aegis transcription acceptance thresholds.
MIN_AVG_LOGPROB = -0.75
MAX_NO_SPEECH_PROB = 0.60
MIN_WORD_PROBABILITY = 0.30
MIN_ACCEPTED_WORD_RATIO = 0.60
MAX_COMPRESSION_RATIO = 2.40


def latest_nearfield_recording():
    files = sorted(
        AUDIO_DIR.glob("nearfield_*.wav")
    )

    if not files:
        return None

    original_files = [
        file
        for file in files
        if "_16k_mono" not in file.stem
        and "_clean" not in file.stem
        and "_trimmed" not in file.stem
        and "_enhanced" not in file.stem
    ]

    if original_files:
        return original_files[-1]

    return files[-1]


def load_model():
    print("===== LOADING FASTER-WHISPER =====")
    print(f"Model: {MODEL_SIZE}")
    print("Device: CPU")
    print("Compute type: int8")

    try:
        model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )

        print("Model loaded.")
        return model

    except Exception as exc:
        print(
            f"\nModel loading failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def calculate_confidence(segment):
    avg_logprob = float(segment.avg_logprob)
    no_speech_prob = float(segment.no_speech_prob)
    compression_ratio = float(segment.compression_ratio)

    word_probabilities = []

    if segment.words:
        for word in segment.words:
            probability = getattr(
                word,
                "probability",
                None,
            )

            if probability is not None:
                word_probabilities.append(
                    float(probability)
                )

    if word_probabilities:
        average_word_probability = (
            sum(word_probabilities)
            / len(word_probabilities)
        )

        accepted_words = sum(
            probability >= MIN_WORD_PROBABILITY
            for probability in word_probabilities
        )

        accepted_word_ratio = (
            accepted_words / len(word_probabilities)
        )

        weakest_word_probability = min(
            word_probabilities
        )
    else:
        average_word_probability = 0.0
        accepted_word_ratio = 0.0
        weakest_word_probability = 0.0

    checks = {
        "avg_logprob": (
            avg_logprob >= MIN_AVG_LOGPROB
        ),
        "no_speech": (
            no_speech_prob <= MAX_NO_SPEECH_PROB
        ),
        "compression": (
            compression_ratio <= MAX_COMPRESSION_RATIO
        ),
        "word_quality": (
            average_word_probability
            >= MIN_WORD_PROBABILITY
        ),
        "word_coverage": (
            accepted_word_ratio
            >= MIN_ACCEPTED_WORD_RATIO
        ),
    }

    passed_checks = sum(
        checks.values()
    )

    confidence_percent = (
        passed_checks
        / len(checks)
        * 100
    )

    if passed_checks == len(checks):
        decision = "ACCEPT"
    elif passed_checks >= 3:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    return {
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
        "compression_ratio": compression_ratio,
        "average_word_probability": (
            average_word_probability
        ),
        "accepted_word_ratio": (
            accepted_word_ratio
        ),
        "weakest_word_probability": (
            weakest_word_probability
        ),
        "passed_checks": passed_checks,
        "total_checks": len(checks),
        "confidence_percent": confidence_percent,
        "decision": decision,
        "checks": checks,
    }


def print_confidence_report(report):
    print("\n===== AEGIS CONFIDENCE REPORT =====")

    print(
        f"Average log probability : "
        f"{report['avg_logprob']:.4f}"
    )

    print(
        f"No-speech probability   : "
        f"{report['no_speech_prob']:.4f}"
    )

    print(
        f"Compression ratio       : "
        f"{report['compression_ratio']:.4f}"
    )

    print(
        f"Average word probability: "
        f"{report['average_word_probability']:.4f}"
    )

    print(
        f"Accepted word ratio     : "
        f"{report['accepted_word_ratio']:.2%}"
    )

    print(
        f"Weakest word probability: "
        f"{report['weakest_word_probability']:.4f}"
    )

    print(
        f"Checks passed           : "
        f"{report['passed_checks']}/"
        f"{report['total_checks']}"
    )

    print(
        f"Confidence              : "
        f"{report['confidence_percent']:.0f}%"
    )

    print(
        f"Decision                : "
        f"{report['decision']}"
    )

    print("\nChecks:")

    for name, passed in report["checks"].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<18} {status}")


def transcribe(model, audio_path):
    print("\n===== TRANSCRIPTION =====")

    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=5,
        temperature=0,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    transcript_parts = []
    segment_reports = []

    for segment in segments:
        text = segment.text.strip()

        if not text:
            continue

        print(
            f"\n[{segment.start:.2f}s -> "
            f"{segment.end:.2f}s]"
        )

        print(f"Text: {text}")

        report = calculate_confidence(
            segment
        )

        segment_reports.append(report)

        print_confidence_report(
            report
        )

        if report["decision"] == "ACCEPT":
            transcript_parts.append(text)

    transcript = " ".join(
        transcript_parts
    ).strip()

    if not transcript:
        transcript = "[NO TRUSTED TRANSCRIPTION]"

    return (
        transcript,
        info,
        segment_reports,
    )


def main():
    print(
        "===== AEGIS FASTER-WHISPER "
        "SMALL.EN CONFIDENCE GATE TEST ====="
    )

    audio_path = latest_nearfield_recording()

    if audio_path is None:
        print("\nNo near-field recording found.")
        print(
            "Expected: "
            "audio\\nearfield_*.wav"
        )
        return

    print("\nAudio:")
    print(audio_path)

    model = load_model()

    if model is None:
        return

    try:
        (
            transcript,
            info,
            reports,
        ) = transcribe(
            model,
            audio_path,
        )

        print("\n===== DETECTED LANGUAGE =====")
        print(f"Language: {info.language}")
        print(
            f"Probability: "
            f"{info.language_probability:.3f}"
        )

        print("\n===== TRUSTED TRANSCRIPT =====")
        print(transcript)

        if reports:
            accepted = sum(
                report["decision"] == "ACCEPT"
                for report in reports
            )

            reviewed = sum(
                report["decision"] == "REVIEW"
                for report in reports
            )

            rejected = sum(
                report["decision"] == "REJECT"
                for report in reports
            )

            print("\n===== AEGIS GATE SUMMARY =====")
            print(f"Accepted segments : {accepted}")
            print(f"Review segments   : {reviewed}")
            print(f"Rejected segments : {rejected}")

            if rejected > 0 and accepted == 0:
                print(
                    "\n⚠️ Aegis would NOT send this "
                    "transcription to the threat detector."
                )
            elif accepted > 0:
                print(
                    "\n✅ Aegis has trusted "
                    "transcription available."
                )
            else:
                print(
                    "\n⚠️ Aegis requires additional "
                    "audio/transcription validation."
                )

        print("\n===== TEST COMPLETE =====")

        print("\nExpected speech:")
        print(
            "Your bank account will be suspended today. "
            "Please give me your OTP to verify your account."
        )

    except Exception as exc:
        print(
            f"\nTranscription failed: "
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()