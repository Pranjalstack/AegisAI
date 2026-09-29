# backend/live_audio.py

import copy
import threading
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

IMMEDIATE_HIGH_SCORE = 8

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
# RUNTIME CONTROL
# ================================================================

_live_thread = None

_live_stop_event = threading.Event()

_live_state_lock = threading.Lock()

_operation_lock = threading.Lock()

_current_operation = None


# ================================================================
# TRUSTED CHUNK HISTORY
# ================================================================

trusted_history = []


# ================================================================
# LIVE STATE
# ================================================================

live_state = {
    "active": False,
    "phase": "STOPPED",
    "chunk_number": 0,
    "completed_chunks": 0,
    "latest_result": None,
    "latest_explanation": "",
    "alert_level": "NO ALERT",
    "alert_reason": "",
    "notification_triggered": False,
    "error": None,
    "started_at": None,
}


# ================================================================
# OPERATION MANAGEMENT
# ================================================================

def try_start_operation(operation_name):
    global _current_operation

    with _operation_lock:
        if _current_operation is not None:
            return False, _current_operation

        _current_operation = operation_name

        return True, None


def finish_operation(operation_name):
    global _current_operation

    with _operation_lock:
        if _current_operation == operation_name:
            _current_operation = None


def get_current_operation():
    with _operation_lock:
        return _current_operation


# ================================================================
# STATE MANAGEMENT
# ================================================================

def reset_live_state():
    with _live_state_lock:
        live_state.clear()

        live_state.update(
            {
                "active": True,
                "phase": "STARTING",
                "chunk_number": 0,
                "completed_chunks": 0,
                "latest_result": None,
                "latest_explanation": "",
                "alert_level": "NO ALERT",
                "alert_reason": "",
                "notification_triggered": False,
                "error": None,
                "started_at": time.time(),
            }
        )


def update_live_state(**values):
    with _live_state_lock:
        for key, value in values.items():
            live_state[key] = value


def get_live_status():
    with _live_state_lock:
        status = copy.deepcopy(
            live_state
        )

    started_at = status.get(
        "started_at"
    )

    if (
        status.get("active")
        and started_at
    ):
        status["uptime_seconds"] = round(
            time.time() - started_at,
            1,
        )
    else:
        status["uptime_seconds"] = 0

    status["operation"] = get_current_operation()

    status["history_size"] = len(
        trusted_history
    )

    return status


# ================================================================
# GENERIC RESULT HELPERS
# ================================================================

def get_field(
    result,
    name,
    default=None,
):
    if result is None:
        return default

    if isinstance(
        result,
        dict,
    ):
        return result.get(
            name,
            default,
        )

    return getattr(
        result,
        name,
        default,
    )


def get_list_value(value):
    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        tuple,
    ):
        return list(value)

    if isinstance(
        value,
        set,
    ):
        return list(value)

    if isinstance(
        value,
        str,
    ):
        value = value.strip()

        if not value:
            return []

        return [value]

    return [str(value)]


def get_score(result):
    value = get_field(
        result,
        "risk_score",
        get_field(
            result,
            "score",
            0,
        ),
    )

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0


def get_risk_level(result):
    value = get_field(
        result,
        "risk_level",
        get_field(
            result,
            "risk",
            "LOW",
        ),
    )

    if value is None:
        return "LOW"

    return str(
        value
    ).upper()


def get_status(result):
    value = get_field(
        result,
        "transcription_status",
        "REJECT",
    )

    if value is None:
        return "REJECT"

    return str(
        value
    ).upper()


def get_confidence(result):
    value = get_field(
        result,
        "transcription_confidence",
        0,
    )

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def get_transcript(result):
    value = get_field(
        result,
        "transcript",
        "",
    )

    if value is None:
        return ""

    return str(
        value
    ).strip()


def get_reasons(result):
    return get_list_value(
        get_field(
            result,
            "reasons",
            [],
        )
    )


def get_categories(result):
    values = get_list_value(
        get_field(
            result,
            "security_categories",
            get_field(
                result,
                "categories",
                [],
            ),
        )
    )

    cleaned = []

    for value in values:
        value = str(
            value
        ).strip().lower()

        if (
            value
            and value not in cleaned
        ):
            cleaned.append(
                value
            )

    return cleaned


# ================================================================
# AUDIO FILE HANDLING
# ================================================================

def get_output_path():
    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    return (
        AUDIO_DIR
        / f"live_chunk_{timestamp}.wav"
    )


def save_audio(
    audio_data,
    path,
):
    audio_array = np.asarray(
        audio_data,
        dtype=np.float32,
    )

    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    audio_array = np.clip(
        audio_array,
        -1.0,
        1.0,
    )

    pcm_data = (
        audio_array * 32767.0
    ).astype(
        np.int16
    )

    with wave.open(
        str(path),
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
            pcm_data.tobytes()
        )


def record_chunk():
    if _live_stop_event.is_set():
        return None

    frames = int(
        SAMPLE_RATE
        * CHUNK_DURATION
    )

    update_live_state(
        phase="RECORDING",
    )

    print(
        "Recording...",
        flush=True,
    )

    audio_data = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=MICROPHONE_DEVICE,
    )

    sd.wait()

    if _live_stop_event.is_set():
        return None

    print(
        "Recording complete.",
        flush=True,
    )

    output_path = get_output_path()

    save_audio(
        audio_data,
        output_path,
    )

    print(
        f"Saved: {output_path}",
        flush=True,
    )

    return output_path


# ================================================================
# TRUST / SUSPICION GATES
# ================================================================

def is_trusted_result(result):
    return (
        get_status(result)
        == "ACCEPT"
    )


def is_suspicious_result(result):
    if not is_trusted_result(
        result
    ):
        return False

    risk_level = get_risk_level(
        result
    )

    score = get_score(
        result
    )

    categories = get_categories(
        result
    )

    if risk_level in {
        "MEDIUM",
        "HIGH",
    }:
        return True

    if score > 0:
        return True

    if categories:
        return True

    return False


# ================================================================
# SINGLE-CHUNK ALERT
# ================================================================

def strong_single_chunk_alert(
    result
):
    if not is_trusted_result(
        result
    ):
        return False

    risk_level = get_risk_level(
        result
    )

    score = get_score(
        result
    )

    if risk_level == "HIGH":
        return True

    if score >= IMMEDIATE_HIGH_SCORE:
        return True

    return False


# ================================================================
# CUMULATIVE EVIDENCE ALERT
# ================================================================

def cumulative_evidence_alert(
    current_result
):
    if not is_suspicious_result(
        current_result
    ):
        return False

    evidence = list(
        trusted_history
    )

    current_score = get_score(
        current_result
    )

    current_categories = get_categories(
        current_result
    )

    evidence.append(
        {
            "risk_score": current_score,
            "categories": current_categories,
            "risk_level": get_risk_level(
                current_result
            ),
        }
    )

    suspicious_chunks = [
        item
        for item in evidence
        if (
            item.get(
                "risk_score",
                0,
            )
            > 0
            or item.get(
                "categories"
            )
            or item.get(
                "risk_level"
            )
            in {
                "MEDIUM",
                "HIGH",
            }
        )
    ]

    if len(
        suspicious_chunks
    ) < MIN_SUSPICIOUS_CHUNKS:
        return False

    total_score = sum(
        int(
            item.get(
                "risk_score",
                0,
            )
        )
        for item in suspicious_chunks
    )

    unique_categories = set()

    for item in suspicious_chunks:
        for category in item.get(
            "categories",
            [],
        ):
            unique_categories.add(
                str(
                    category
                ).lower()
            )

    has_critical_category = bool(
        unique_categories.intersection(
            CRITICAL_CATEGORIES
        )
    )

    if (
        total_score
        >= CUMULATIVE_SCORE_THRESHOLD
        and len(
            unique_categories
        )
        >= MIN_UNIQUE_CATEGORIES
        and has_critical_category
    ):
        return True

    return False


# ================================================================
# CORROBORATION POLICY
# ================================================================

def corroborated_alert(
    current_result
):
    if not is_suspicious_result(
        current_result
    ):
        return False

    current_categories = set(
        get_categories(
            current_result
        )
    )

    if not current_categories:
        return False

    previous_categories = set()

    for item in trusted_history:
        for category in item.get(
            "categories",
            [],
        ):
            previous_categories.add(
                str(
                    category
                ).lower()
            )

    overlap = (
        current_categories.intersection(
            previous_categories
        )
    )

    if (
        overlap
        and get_risk_level(
            current_result
        )
        in {
            "MEDIUM",
            "HIGH",
        }
    ):
        return True

    return False


# ================================================================
# ALERT DECISION ENGINE
# ================================================================

def determine_alert_level(
    result
):
    if not is_trusted_result(
        result
    ):
        return (
            "NO ALERT",
            "Transcription was not trusted.",
        )

    if strong_single_chunk_alert(
        result
    ):
        return (
            "ALERT",
            "Strong high-risk evidence detected in one trusted chunk.",
        )

    if cumulative_evidence_alert(
        result
    ):
        return (
            "ALERT",
            "Multiple trusted security indicators met the cumulative alert policy.",
        )

    if corroborated_alert(
        result
    ):
        return (
            "REVIEW",
            "Suspicious evidence was corroborated across trusted chunks.",
        )

    risk_level = get_risk_level(
        result
    )

    if risk_level in {
        "MEDIUM",
        "HIGH",
    }:
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

def store_trusted_result(
    result
):
    if not is_trusted_result(
        result
    ):
        return

    entry = {
        "risk_score": get_score(
            result
        ),
        "risk_level": get_risk_level(
            result
        ),
        "categories": get_categories(
            result
        ),
        "timestamp": time.time(),
    }

    trusted_history.append(
        entry
    )

    while len(
        trusted_history
    ) > HISTORY_SIZE:
        trusted_history.pop(0)


def get_history_summary():
    total_score = 0
    categories = set()

    for item in trusted_history:
        total_score += int(
            item.get(
                "risk_score",
                0,
            )
        )

        for category in item.get(
            "categories",
            [],
        ):
            categories.add(
                str(
                    category
                ).lower()
            )

    return {
        "stored_chunks": len(
            trusted_history
        ),
        "cumulative_score": total_score,
        "categories": sorted(
            categories
        ),
    }


# ================================================================
# RESULT SERIALIZATION FOR API
# ================================================================

def serialize_result(
    result
):
    if result is None:
        return None

    return {
        "risk": get_risk_level(
            result
        ),
        "risk_level": get_risk_level(
            result
        ),
        "score": get_score(
            result
        ),
        "risk_score": get_score(
            result
        ),
        "transcription_status": get_status(
            result
        ),
        "transcription_confidence": get_confidence(
            result
        ),
        "transcript": get_transcript(
            result
        ),
        "analysis_transcript": str(
            get_field(
                result,
                "analysis_transcript",
                "",
            )
            or ""
        ),
        "reasons": get_reasons(
            result
        ),
        "urls": get_list_value(
            get_field(
                result,
                "urls",
                [],
            )
        ),
        "security_categories": get_categories(
            result
        ),
        "transcription_metrics": get_field(
            result,
            "transcription_metrics",
            {},
        ),
        "confidence_checks": get_field(
            result,
            "confidence_checks",
            {},
        ),
    }


# ================================================================
# TERMINAL OUTPUT
# ================================================================

def print_result(
    chunk_number,
    path,
    result,
    explanation,
    alert_level,
    alert_reason,
):
    risk_level = get_risk_level(
        result
    )

    risk_score = get_score(
        result
    )

    status = get_status(
        result
    )

    confidence = get_confidence(
        result
    )

    transcript = get_transcript(
        result
    )

    reasons = get_reasons(
        result
    )

    categories = get_categories(
        result
    )

    print()
    print("=" * 64)
    print(
        f"AEGIS LIVE CHUNK #{chunk_number}"
    )
    print(
        f"File: {path.name}"
    )
    print(
        f"Risk Level: {risk_level}"
    )
    print(
        f"Risk Score: {risk_score}"
    )
    print(
        f"Transcription Status: {status}"
    )
    print(
        f"Transcription Confidence: "
        f"{confidence:.0f}%"
    )

    print()
    print("Transcript:")

    if transcript:
        print(
            transcript
        )
    else:
        print(
            "[No trusted transcription]"
        )

    print()
    print("Detected Indicators:")

    if reasons:
        for reason in reasons:
            print(
                f"• {reason}"
            )
    else:
        print(
            "• None"
        )

    print()
    print("Security Categories:")

    if categories:
        for category in categories:
            print(
                f"• {category}"
            )
    else:
        print(
            "• None"
        )

    print()
    print("Alert Decision:")

    if alert_level == "ALERT":
        print(
            "🚨 AEGIS ALERT"
        )

        print()
        print(
            "!!! SECURITY WARNING !!!"
        )

        print(
            alert_reason
        )

    elif alert_level == "REVIEW":
        print(
            "⚠️ REVIEW"
        )

        print(
            alert_reason
        )

    else:
        print(
            "No alert"
        )

    print()
    print("AI Explanation:")

    if explanation:
        print(
            explanation
        )
    else:
        print(
            "[No AI explanation available]"
        )

    print("=" * 64)


# ================================================================
# CHUNK ANALYSIS
# ================================================================

def analyze_chunk(
    path,
    chunk_number,
):
    update_live_state(
        phase="ANALYZING",
        chunk_number=chunk_number,
        error=None,
    )

    print()
    print(
        f"[Chunk {chunk_number}] "
        "Running local Aegis analysis...",
        flush=True,
    )

    try:
        result, explanation = (
            analyze_audio_with_ai(
                str(path)
            )
        )

    except Exception as error:
        error_message = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        update_live_state(
            phase="ERROR",
            chunk_number=chunk_number,
            error=error_message,
            latest_result=None,
            latest_explanation="",
            alert_level="NO ALERT",
            alert_reason="Analysis failed.",
            notification_triggered=False,
        )

        print()
        print(
            f"[Chunk {chunk_number}] "
            f"Analysis error: {error_message}",
            flush=True,
        )

        return

    alert_level, alert_reason = (
        determine_alert_level(
            result
        )
    )

    notification_triggered = False

    if alert_level == "ALERT":
        notification_triggered = (
            notify_security_alert(
                risk_level=get_risk_level(
                    result
                ),
                risk_score=get_score(
                    result
                ),
                transcript=get_transcript(
                    result
                ),
                categories=get_categories(
                    result
                ),
                reasons=get_reasons(
                    result
                ),
            )
        )

    serialized_result = (
        serialize_result(
            result
        )
    )

    store_trusted_result(
        result
    )

    history_summary = (
        get_history_summary()
    )

    update_live_state(
        phase=(
            "ALERT"
            if alert_level == "ALERT"
            else "READY"
        ),
        chunk_number=chunk_number,
        completed_chunks=chunk_number,
        latest_result=serialized_result,
        latest_explanation=(
            explanation
            or ""
        ),
        alert_level=alert_level,
        alert_reason=alert_reason,
        notification_triggered=(
            notification_triggered
        ),
        error=None,
        history_summary=history_summary,
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
        print()
        print(
            "!" * 64
        )

        print(
            "🚨 AEGIS SECURITY ALERT"
        )

        print(
            alert_reason
        )

        if notification_triggered:
            print(
                "Desktop security notification triggered."
            )
        else:
            print(
                "Desktop notification suppressed by alert cooldown."
            )

        print(
            "!" * 64
        )

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

    print()
    print(
        "Recent trusted chunks stored in memory: "
        f"{len(trusted_history)}"
    )


# ================================================================
# LIVE GUARD LOOP
# ================================================================

def run_live_guard():
    global _live_thread

    chunk_number = 1

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

    try:
        while not _live_stop_event.is_set():
            print()
            print(
                "-" * 64
            )

            print(
                f"Preparing audio chunk #{chunk_number}..."
            )

            path = record_chunk()

            if path is None:
                break

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
                except Exception as error:
                    print(
                        f"Could not delete temporary audio: {error}"
                    )

            chunk_number += 1

            if not _live_stop_event.is_set():
                print()
                print(
                    "Starting next chunk..."
                )

    except Exception as error:
        error_message = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        update_live_state(
            phase="ERROR",
            error=error_message,
        )

        print()
        print(
            f"Aegis live guard error: "
            f"{error_message}",
            flush=True,
        )

    finally:
        try:
            sd.stop()
        except Exception:
            pass

        update_live_state(
            active=False,
            phase="STOPPED",
            error=live_state.get(
                "error"
            ),
        )

        finish_operation(
            "live-audio"
        )

        _live_thread = None

        print()
        print(
            "=" * 64
        )
        print(
            "Aegis real-time audio guard stopped."
        )
        print(
            "=" * 64
        )


# ================================================================
# START / STOP API
# ================================================================

def start_live_guard():
    global _live_thread

    if (
        _live_thread is not None
        and _live_thread.is_alive()
    ):
        return {
            "started": False,
            "message": (
                "Live Audio Guard is already running."
            ),
        }

    acquired, existing = (
        try_start_operation(
            "live-audio"
        )
    )

    if not acquired:
        return {
            "started": False,
            "message": (
                "Aegis is currently busy with "
                f"{existing}."
            ),
        }

    trusted_history.clear()

    _live_stop_event.clear()

    reset_live_state()

    _live_thread = threading.Thread(
        target=run_live_guard,
        name="AegisLiveAudioGuard",
        daemon=True,
    )

    _live_thread.start()

    return {
        "started": True,
        "message": (
            "Aegis Live Audio Guard started."
        ),
    }


def stop_live_guard():
    if (
        _live_thread is None
        or not _live_thread.is_alive()
    ):
        update_live_state(
            active=False,
            phase="STOPPED",
        )

        finish_operation(
            "live-audio"
        )

        return {
            "stopped": False,
            "message": (
                "Live Audio Guard is not running."
            ),
        }

    _live_stop_event.set()

    update_live_state(
        phase="STOPPING",
    )

    return {
        "stopped": True,
        "message": (
            "Stop requested. "
            "Aegis will finish the current recording operation."
        ),
    }


# ================================================================
# TERMINAL ENTRY POINT
# ================================================================

def main():
    result = start_live_guard()

    print(
        result["message"]
    )

    if not result["started"]:
        return

    print(
        "Press Ctrl+C to stop Aegis."
    )

    try:
        while (
            _live_thread is not None
            and _live_thread.is_alive()
        ):
            time.sleep(0.5)

    except KeyboardInterrupt:
        print()
        print(
            "Stopping Aegis..."
        )

        stop_live_guard()

        while (
            _live_thread is not None
            and _live_thread.is_alive()
        ):
            time.sleep(0.2)


if __name__ == "__main__":
    main()