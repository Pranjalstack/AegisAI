# backend/audio_analyzer.py

from pathlib import Path
from collections import Counter
import re

from faster_whisper import WhisperModel

from backend.threat_detector import analyze_text
from backend.local_ai import explain_threat


MODEL_NAME = "small.en"

MIN_AVG_LOGPROB = -0.75
MAX_NO_SPEECH_PROB = 0.60
MIN_AVG_WORD_PROBABILITY = 0.30
MIN_ACCEPTED_WORD_RATIO = 0.60
MAX_COMPRESSION_RATIO = 2.40
MAX_REPETITION_RATIO = 0.50


print(
    "Loading Aegis local Faster-Whisper speech model...",
    flush=True,
)


def load_whisper_model():
    try:
        print(
            "Trying Faster-Whisper on GPU...",
            flush=True,
        )

        model = WhisperModel(
            MODEL_NAME,
            device="cuda",
            compute_type="float16",
        )

        print(
            "Aegis Faster-Whisper GPU model loaded.",
            flush=True,
        )

        return model, "cuda"

    except Exception as gpu_error:
        print(
            "GPU Faster-Whisper unavailable.",
            flush=True,
        )

        print(
            f"GPU reason: {type(gpu_error).__name__}: "
            f"{gpu_error}",
            flush=True,
        )

        print(
            "Falling back to local CPU Faster-Whisper...",
            flush=True,
        )

        model = WhisperModel(
            MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )

        print(
            "Aegis Faster-Whisper CPU model loaded.",
            flush=True,
        )

        return model, "cpu"


speech_model, speech_device = load_whisper_model()


def repetition_ratio(text):
    words = re.findall(
        r"[A-Za-z0-9']+",
        text.lower(),
    )

    if len(words) < 6:
        return 0.0

    counts = Counter(words)

    repeated_words = sum(
        count
        for count in counts.values()
        if count > 1
    )

    return repeated_words / len(words)


def repeated_phrase_detected(text):
    words = re.findall(
        r"[A-Za-z0-9']+",
        text.lower(),
    )

    if len(words) < 8:
        return False

    for phrase_length in (2, 3, 4, 5):
        if len(words) < phrase_length * 3:
            continue

        phrases = []

        for index in range(
            len(words) - phrase_length + 1
        ):
            phrase = tuple(
                words[
                    index:index + phrase_length
                ]
            )

            phrases.append(phrase)

        counts = Counter(phrases)

        if not counts:
            continue

        _, count = counts.most_common(1)[0]

        if count >= 4:
            return True

    return False


def calculate_confidence(segment, text):
    avg_logprob = float(
        getattr(
            segment,
            "avg_logprob",
            -99.0,
        )
    )

    no_speech_prob = float(
        getattr(
            segment,
            "no_speech_prob",
            1.0,
        )
    )

    compression_ratio = float(
        getattr(
            segment,
            "compression_ratio",
            99.0,
        )
    )

    probabilities = []

    if segment.words:
        for word in segment.words:
            probability = getattr(
                word,
                "probability",
                None,
            )

            if probability is not None:
                probabilities.append(
                    float(probability)
                )

    if probabilities:
        average_word_probability = (
            sum(probabilities)
            / len(probabilities)
        )

        accepted_words = sum(
            probability >= MIN_AVG_WORD_PROBABILITY
            for probability in probabilities
        )

        accepted_word_ratio = (
            accepted_words
            / len(probabilities)
        )

        weakest_word_probability = min(
            probabilities
        )

    else:
        average_word_probability = 0.0
        accepted_word_ratio = 0.0
        weakest_word_probability = 0.0

    repetition = repetition_ratio(text)

    repeated_phrase = repeated_phrase_detected(
        text
    )

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
        "word_probability": (
            average_word_probability
            >= MIN_AVG_WORD_PROBABILITY
        ),
        "word_coverage": (
            accepted_word_ratio
            >= MIN_ACCEPTED_WORD_RATIO
        ),
        "repetition": (
            repetition <= MAX_REPETITION_RATIO
            and not repeated_phrase
        ),
    }

    passed = sum(
        checks.values()
    )

    total = len(checks)

    confidence_percent = (
        passed / total * 100
    )

    if passed == total:
        decision = "ACCEPT"

    elif passed >= 4:
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
        "repetition_ratio": repetition,
        "repeated_phrase_detected": (
            repeated_phrase
        ),
        "passed_checks": passed,
        "total_checks": total,
        "confidence_percent": confidence_percent,
        "decision": decision,
        "checks": checks,
    }


def transcribe_audio(
    audio_path,
    return_metadata=False,
):
    segments, info = speech_model.transcribe(
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

        transcript_parts.append(text)

        report = calculate_confidence(
            segment,
            text,
        )

        segment_reports.append(
            report
        )

    transcript = " ".join(
        transcript_parts
    ).strip()

    if not transcript:
        metadata = {
            "analysis_status": "REJECT",
            "reason": "No speech was transcribed.",
            "device": speech_device,
            "model": MODEL_NAME,
            "segments": [],
        }

        if return_metadata:
            return "", metadata

        return ""

    if not segment_reports:
        metadata = {
            "analysis_status": "REJECT",
            "reason": (
                "No usable transcription segment "
                "was produced."
            ),
            "device": speech_device,
            "model": MODEL_NAME,
            "segments": [],
        }

        if return_metadata:
            return transcript, metadata

        return transcript

    accepted_segments = sum(
        report["decision"] == "ACCEPT"
        for report in segment_reports
    )

    review_segments = sum(
        report["decision"] == "REVIEW"
        for report in segment_reports
    )

    rejected_segments = sum(
        report["decision"] == "REJECT"
        for report in segment_reports
    )

    confidence_values = [
        report["confidence_percent"]
        for report in segment_reports
    ]

    overall_confidence = (
        sum(confidence_values)
        / len(confidence_values)
    )

    if rejected_segments > 0:
        overall_decision = "REJECT"

    elif review_segments > 0:
        overall_decision = "REVIEW"

    else:
        overall_decision = "ACCEPT"

    metadata = {
        "analysis_status": overall_decision,
        "confidence_percent": overall_confidence,
        "device": speech_device,
        "model": MODEL_NAME,
        "accepted_segments": accepted_segments,
        "review_segments": review_segments,
        "rejected_segments": rejected_segments,
        "segments": segment_reports,
    }

    if return_metadata:
        return transcript, metadata

    return transcript


def scan_audio(audio_path):
    text, metadata = transcribe_audio(
        audio_path,
        return_metadata=True,
    )

    if not text:
        return {
            "text": "",
            "risk": "LOW",
            "score": 0,
            "reasons": [
                "No usable speech could be transcribed."
            ],
            "urls": [],
            "metadata": metadata,
        }

    status = metadata.get(
        "analysis_status",
        "REJECT",
    )

    if status != "ACCEPT":
        confidence = metadata.get(
            "confidence_percent",
            0,
        )

        if status == "REVIEW":
            reason = (
                "Speech transcription confidence "
                f"is insufficient for reliable "
                f"security analysis "
                f"({confidence:.0f}%)."
            )

        else:
            reason = (
                "Speech transcription was rejected "
                "because the audio produced "
                "unreliable or repetitive text."
            )

        return {
            "text": text,
            "risk": "LOW",
            "score": 0,
            "reasons": [
                reason
            ],
            "urls": [],
            "metadata": metadata,
        }

    result = analyze_text(
        text
    )

    result["text"] = text
    result["metadata"] = metadata

    return result


def analyze_audio_with_ai(audio_path):
    result = scan_audio(
        audio_path
    )

    metadata = result.get(
        "metadata",
        {},
    )

    status = metadata.get(
        "analysis_status",
        "REJECT",
    )

    text = result.get(
        "text",
        "",
    ).strip()

    if status != "ACCEPT":
        confidence = metadata.get(
            "confidence_percent",
            0,
        )

        if text:
            explanation = (
                "The speech transcript was not "
                "reliable enough for security analysis. "
                f"Transcription confidence was "
                f"{confidence:.0f}%. "
                "Aegis did not use the uncertain "
                "transcript to determine a threat."
            )

        else:
            explanation = (
                "No reliable speech transcription "
                "was available for security analysis."
            )

        return {
            "scan": result,
            "explanation": explanation,
        }

    explanation = explain_threat(
        text,
        result,
    )

    return {
        "scan": result,
        "explanation": explanation,
    }