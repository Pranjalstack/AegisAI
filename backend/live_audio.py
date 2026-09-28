import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from backend.alert_notifier import notify_security_alert
from backend.audio_analyzer import analyze_audio_with_ai


# ================================================================
# AEGIS REAL-TIME AUDIO CONFIGURATION
# ================================================================

AUDIO_DIR = Path("audio")

MICROPHONE_DEVICE = 1
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK_DURATION = 8

KEEP_RECORDINGS = False

HISTORY_SIZE = 3

# Immediate single-chunk alert.
IMMEDIATE_HIGH_SCORE = 8

# Cumulative evidence alert.
CUMULATIVE_SCORE_THRESHOLD = 5
MIN_SUSPICIOUS_CHUNKS = 2
MIN_UNIQUE_CATEGORIES = 2

CRITICAL_CATEGORIES = {
    "credentials",
    "payment",
    "urgency",
    "impersonation",
}


# ================================================================
# RECENT TRUSTED CHUNK MEMORY
# ================================================================

trusted_history = []


# ================================================================
# GENERIC RESULT ACCESS
# Supports dictionaries and dataclass/object results.
# ================================================================

def get_field(result, name, default=None):
    if result is None:
        return default

    if isinstance(result, dict):
        return result.get(name, default)

    return getattr(result, name, default)


def get_list_value(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        return [value]

    return [str(value)]


def get_score(result):
    value = get_field(result, "risk_score", 0)

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def get_risk_level(result):
    value = get_field(result, "risk_level", "LOW")

    if value is None:
        return "LOW"

    return str(value).upper()


def get_status(result):
    value = get_field(result, "transcription_status", "REJECT")

    if value is None:
        return "REJECT"

    return str(value).upper()


def get_confidence(result):
    value = get_field(result, "transcription_confidence", 0)

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def get_transcript(result):
    value = get_field(result, "transcript", "")

    if value is None:
        return ""

    return str(value).strip()


def get_reasons(result):
    return get_list_value(
        get_field(result, "reasons", [])
    )


def get_categories(result):
    values = get_list_value(
        get_field(result, "security_categories", [])
    )

    cleaned = []

    for value in values:
        value = str(value).strip().lower()

        if value and value not in cleaned:
            cleaned.append(value)

    return cleaned


# ================================================================
# AUDIO FILE HANDLING
# ================================================================

def get_output_path():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    return AUDIO_DIR / f"live_chunk_{timestamp}.wav"


def save_audio(audio_data, path):
    audio_array = np.asarray(audio_data, dtype=np.float32)

    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    audio_array = np.clip(audio_array, -1.0, 1.0)

    pcm_data = (
        audio_array * 32767.0
    ).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_data.tobytes())


def record_chunk():
    frames = int(
        SAMPLE_RATE * CHUNK_DURATION
    )

    print("Recording...")

    audio_data = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=MICROPHONE_DEVICE,
    )

    sd.wait()

    print("Recording complete.")

    output_path = get_output_path()

    save_audio(
        audio_data,
        output_path,
    )

    print(f"Saved: {output_path}")

    return output_path


# ================================================================
# TRUST / SUSPICION GATES
# ================================================================

def is_trusted_result(result):
    return get_status(result) == "ACCEPT"


def is_suspicious_result(result):
    if not is_trusted_result(result):
        return False

    risk_level = get_risk_level(result)
    score = get_score(result)
    categories = get_categories(result)

    if risk_level in {"MEDIUM", "HIGH"}:
        return True

    if score > 0:
        return True

    if categories:
        return True

    return False


# ================================================================
# SINGLE CHUNK ALERT
# ================================================================

def strong_single_chunk_alert(result):
    if not is_trusted_result(result):
        return False

    risk_level = get_risk_level(result)
    score = get_score(result)

    if risk_level == "HIGH":
        return True

    if score >= IMMEDIATE_HIGH_SCORE:
        return True

    return False


# ================================================================
# CUMULATIVE EVIDENCE ALERT
# ================================================================

def cumulative_evidence_alert(current_result):
    if not is_suspicious_result(current_result):
        return False

    evidence = list(trusted_history)

    current_score = get_score(current_result)
    current_categories = get_categories(current_result)

    evidence.append(
        {
            "risk_score": current_score,
            "categories": current_categories,
            "risk_level": get_risk_level(current_result),
        }
    )

    suspicious_chunks = [
        item
        for item in evidence
        if item.get("risk_score", 0) > 0
        or item.get("categories")
        or item.get("risk_level") in {"MEDIUM", "HIGH"}
    ]

    if len(suspicious_chunks) < MIN_SUSPICIOUS_CHUNKS:
        return False

    total_score = sum(
        int(item.get("risk_score", 0))
        for item in suspicious_chunks
    )

    unique_categories = set()

    for item in suspicious_chunks:
        for category in item.get("categories", []):
            unique_categories.add(
                str(category).lower()
            )

    has_critical_category = bool(
        unique_categories.intersection(
            CRITICAL_CATEGORIES
        )
    )

    if (
        total_score >= CUMULATIVE_SCORE_THRESHOLD
        and len(unique_categories) >= MIN_UNIQUE_CATEGORIES
        and has_critical_category
    ):
        return True

    return False


# ================================================================
# CORROBORATION POLICY
# ================================================================

def corroborated_alert(current_result):
    if not is_suspicious_result(current_result):
        return False

    current_categories = set(
        get_categories(current_result)
    )

    if not current_categories:
        return False

    previous_categories = set()

    for item in trusted_history:
        for category in item.get("categories", []):
            previous_categories.add(
                str(category).lower()
            )

    overlap = current_categories.intersection(
        previous_categories
    )

    if overlap and get_risk_level(current_result) in {
        "MEDIUM",
        "HIGH",
    }:
        return True

    return False


# ================================================================
# ALERT DECISION ENGINE
# ================================================================

def determine_alert_level(result):
    if not is_trusted_result(result):
        return (
            "NO ALERT",
            "Transcription was not trusted.",
        )

    if strong_single_chunk_alert(result):
        return (
            "ALERT",
            "Strong high-risk evidence detected in one trusted chunk.",
        )

    if cumulative_evidence_alert(result):
        return (
            "ALERT",
            "Multiple trusted security indicators met the cumulative alert policy.",
        )

    if corroborated_alert(result):
        return (
            "REVIEW",
            "Suspicious evidence was corroborated across trusted chunks.",
        )

    risk_level = get_risk_level(result)

    if risk_level in {"MEDIUM", "HIGH"}:
        return (
            "REVIEW",
            "A trusted chunk contains elevated security risk.",
        )

    return (
        "NO ALERT",
        "No alert threshold was reached.",
    )


# ================================================================
# HISTORY
# ================================================================

def store_trusted_result(result):
    if not is_trusted_result(result):
        return

    entry = {
        "risk_score": get_score(result),
        "risk_level": get_risk_level(result),
        "categories": get_categories(result),
        "timestamp": time.time(),
    }

    trusted_history.append(entry)

    while len(trusted_history) > HISTORY_SIZE:
        trusted_history.pop(0)


# ================================================================
# OUTPUT
# ================================================================

def print_result(
    chunk_number,
    path,
    result,
    explanation,
    alert_level,
    alert_reason,
):
    risk_level = get_risk_level(result)
    risk_score = get_score(result)
    status = get_status(result)
    confidence = get_confidence(result)
    transcript = get_transcript(result)
    reasons = get_reasons(result)
    categories = get_categories(result)

    print()
    print("=" * 64)
    print(f"AEGIS LIVE CHUNK #{chunk_number}")
    print(f"File: {path.name}")
    print(f"Risk Level: {risk_level}")
    print(f"Risk Score: {risk_score}")
    print(f"Transcription Status: {status}")
    print(f"Transcription Confidence: {confidence:.0f}%")
    print()
    print("Transcript:")

    if transcript:
        print(transcript)
    else:
        print("[No trusted transcription]")

    print()
    print("Detected Indicators:")

    if reasons:
        for reason in reasons:
            print(f"• {reason}")
    else:
        print("• None")

    print()
    print("Security Categories:")

    if categories:
        for category in categories:
            print(f"• {category}")
    else:
        print("• None")

    print()
    print("Alert Decision:")

    if alert_level == "ALERT":
        print("🚨 AEGIS ALERT")

        print()
        print("!!! SECURITY WARNING !!!")
        print(alert_reason)

    elif alert_level == "REVIEW":
        print("⚠️ REVIEW")
        print(alert_reason)

    else:
        print("No alert")

    print()
    print("AI Explanation:")

    if explanation:
        print(explanation)
    else:
        print("[No AI explanation available]")

    print("=" * 64)


# ================================================================
# CHUNK ANALYSIS
# ================================================================

def analyze_chunk(path, chunk_number):
    print()
    print(
        f"[Chunk {chunk_number}] "
        "Running local Aegis analysis..."
    )

    try:
        result, explanation = analyze_audio_with_ai(
            str(path)
        )

    except Exception as exc:
        print()
        print(
            f"[Chunk {chunk_number}] "
            f"Analysis error: {exc}"
        )

        return

    alert_level, alert_reason = determine_alert_level(
        result
    )

    print_result(
        chunk_number=chunk_number,
        path=path,
        result=result,
        explanation=explanation,
        alert_level=alert_level,
        alert_reason=alert_reason,
    )

    if alert_level == "ALERT":
        risk_level = get_risk_level(result)
        risk_score = get_score(result)
        transcript = get_transcript(result)
        categories = get_categories(result)
        reasons = get_reasons(result)

        notification_triggered = notify_security_alert(
            risk_level=risk_level,
            risk_score=risk_score,
            transcript=transcript,
            categories=categories,
            reasons=reasons,
        )

        print()
        print("!" * 64)
        print("🚨 AEGIS SECURITY ALERT")
        print(alert_reason)

        if notification_triggered:
            print(
                "Desktop security notification triggered."
            )
        else:
            print(
                "Desktop notification suppressed by alert cooldown."
            )

        print("!" * 64)

    elif alert_level == "REVIEW":
        print()
        print(
            "Aegis: Suspicious activity requires review."
        )

    else:
        print()
        print(
            "Aegis: No alert triggered."
        )

    store_trusted_result(result)

    print()
    print(
        "Recent trusted chunks stored in memory: "
        f"{len(trusted_history)}"
    )


# ================================================================
# MAIN LOOP
# ================================================================

def main():
    print("=" * 64)
    print(
        "              AEGIS REAL-TIME AUDIO GUARD"
    )
    print("=" * 64)

    print(
        f"Microphone device : {MICROPHONE_DEVICE}"
    )

    print(
        f"Sample rate       : {SAMPLE_RATE} Hz"
    )

    print(
        f"Channels          : {CHANNELS}"
    )

    print(
        f"Chunk duration    : {CHUNK_DURATION} sec"
    )

    print(
        f"Keep recordings   : {KEEP_RECORDINGS}"
    )

    print()
    print("Pipeline:")
    print(
        "Microphone -> Faster-Whisper small.en "
        "-> Confidence Gate -> Security Detector "
        "-> Cumulative Evidence -> Local Qwen "
        "-> Alert Policy -> Desktop Notification"
    )

    print()
    print("Alert system:")
    print(
        "• Untrusted transcription can never trigger an alert."
    )
    print(
        "• Strong HIGH-risk evidence may alert immediately."
    )
    print(
        "• Multiple trusted suspicious chunks can combine into one alert."
    )
    print(
        "• Desktop alerts have a cooldown to prevent notification spam."
    )

    print()
    print(
        "Continuous monitoring is active."
    )

    print(
        "Press Ctrl+C to stop Aegis."
    )

    print()

    chunk_number = 1

    try:
        while True:
            print("-" * 64)

            print(
                f"Preparing audio chunk #{chunk_number}..."
            )

            try:
                path = record_chunk()

            except Exception as exc:
                print()
                print(
                    f"Microphone recording error: {exc}"
                )

                print(
                    "Retrying in 2 seconds..."
                )

                time.sleep(2)

                continue

            analyze_chunk(
                path=path,
                chunk_number=chunk_number,
            )

            if (
                not KEEP_RECORDINGS
                and path.exists()
            ):
                try:
                    path.unlink()

                except Exception as exc:
                    print(
                        f"Could not delete temporary audio: {exc}"
                    )

            chunk_number += 1

            print()
            print("Starting next chunk...")

    except KeyboardInterrupt:
        print()
        print("=" * 64)
        print(
            "Aegis real-time audio guard stopped."
        )
        print("=" * 64)

    finally:
        try:
            sd.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()