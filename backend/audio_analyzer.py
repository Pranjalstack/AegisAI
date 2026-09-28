# backend/audio_analyzer.py

import re
import statistics
from pathlib import Path

from faster_whisper import WhisperModel

from backend.threat_detector import analyze_text
from backend.local_ai import explain_threat


# ================================================================
# MODEL CONFIGURATION
# ================================================================

MODEL_NAME = "small.en"

GPU_DEVICE = "cuda"
GPU_COMPUTE_TYPE = "float16"

CPU_DEVICE = "cpu"
CPU_COMPUTE_TYPE = "int8"


# ================================================================
# CONFIDENCE GATE CONFIGURATION
# ================================================================

MIN_AVG_LOGPROB = -0.75
MAX_NO_SPEECH_PROB = 0.60
MIN_AVG_WORD_PROBABILITY = 0.30
MIN_ACCEPTED_WORD_RATIO = 0.60
MAX_COMPRESSION_RATIO = 2.40
MAX_REPETITION_RATIO = 0.50

MIN_ACCEPTED_CHECKS = 4
MIN_REVIEW_CHECKS = 3


# ================================================================
# SECURITY TRANSCRIPT NORMALIZATION
# ================================================================

SECURITY_CONTEXT_PATTERN = re.compile(
    r"\b(?:bank|account|verify|verification|password|otp|one[-\s]?time|"
    r"pin|cvv|security|code|payment|card|upi|ifsc|suspended|blocked|"
    r"locked|transfer|money|fee)\b",
    re.IGNORECASE,
)


def normalize_security_transcript(text):
    """
    Preserve the original transcript while creating an
    analysis-only normalized copy for common STT variations.
    """

    text = str(text or "")

    if not text.strip():
        return ""

    normalized = text

    security_context_present = bool(
        SECURITY_CONTEXT_PATTERN.search(
            normalized
        )
    )

    if security_context_present:
        normalized = re.sub(
            r"\br\s*o\s*t\s*p\b",
            "otp",
            normalized,
            flags=re.IGNORECASE,
        )

        normalized = re.sub(
            r"\brtp\b",
            "otp",
            normalized,
            flags=re.IGNORECASE,
        )

        normalized = re.sub(
            r"\br\s*t\s*p\b",
            "otp",
            normalized,
            flags=re.IGNORECASE,
        )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


# ================================================================
# MODEL LOADING
# ================================================================

print(
    "Loading Aegis local Faster-Whisper speech model...",
    flush=True,
)

speech_model = None

try:
    print(
        "Trying Faster-Whisper on GPU...",
        flush=True,
    )

    speech_model = WhisperModel(
        MODEL_NAME,
        device=GPU_DEVICE,
        compute_type=GPU_COMPUTE_TYPE,
    )

    print(
        "Aegis Faster-Whisper GPU model loaded.",
        flush=True,
    )

except Exception as gpu_error:
    print(
        f"GPU speech model failed: {gpu_error}",
        flush=True,
    )

    print(
        "Falling back to Faster-Whisper CPU...",
        flush=True,
    )

    speech_model = WhisperModel(
        MODEL_NAME,
        device=CPU_DEVICE,
        compute_type=CPU_COMPUTE_TYPE,
    )

    print(
        "Aegis Faster-Whisper CPU model loaded.",
        flush=True,
    )


# ================================================================
# TEXT HELPERS
# ================================================================

def clean_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()


def tokenise(text):
    return re.findall(
        r"[A-Za-z0-9']+",
        str(text or "").lower(),
    )


def calculate_repetition_ratio(text):
    """
    Estimate repetitive speech using repeated adjacent
    tokens and repeated bigrams.
    """

    tokens = tokenise(text)

    if len(tokens) < 4:
        return 0.0

    repeated_token_count = 0

    for index in range(1, len(tokens)):
        if tokens[index] == tokens[index - 1]:
            repeated_token_count += 1

    repeated_bigram_count = 0

    bigrams = [
        tuple(tokens[index:index + 2])
        for index in range(
            0,
            len(tokens) - 1,
        )
    ]

    if bigrams:
        counts = {}

        for bigram in bigrams:
            counts[bigram] = (
                counts.get(
                    bigram,
                    0,
                )
                + 1
            )

        for count in counts.values():
            if count >= 2:
                repeated_bigram_count += (
                    count - 1
                )

    token_ratio = (
        repeated_token_count
        / len(tokens)
    )

    bigram_ratio = 0.0

    if bigrams:
        bigram_ratio = (
            repeated_bigram_count
            / len(bigrams)
        )

    return max(
        token_ratio,
        bigram_ratio,
    )


def safe_average(
    values,
    default=0.0,
):
    cleaned = []

    for value in values:
        try:
            numeric = float(value)

            if numeric == numeric:
                cleaned.append(
                    numeric
                )

        except (
            TypeError,
            ValueError,
        ):
            continue

    if not cleaned:
        return default

    return statistics.mean(
        cleaned
    )


# ================================================================
# WHISPER TRANSCRIPTION
# ================================================================

def transcribe_audio(audio_path):
    """
    Transcribe an audio file using Faster-Whisper and calculate
    confidence metrics for the Aegis confidence gate.
    """

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file does not exist: {audio_path}"
        )

    segments_generator, info = speech_model.transcribe(
        str(path),
        language="en",
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
        },
        word_timestamps=True,
    )

    segments = list(
        segments_generator
    )

    segment_texts = []

    avg_logprobs = []
    no_speech_probs = []
    compression_ratios = []
    word_probabilities = []

    total_words = 0
    accepted_words = 0

    for segment in segments:
        segment_text = clean_text(
            getattr(
                segment,
                "text",
                "",
            )
        )

        if segment_text:
            segment_texts.append(
                segment_text
            )

        avg_logprobs.append(
            getattr(
                segment,
                "avg_logprob",
                -10.0,
            )
        )

        no_speech_probs.append(
            getattr(
                segment,
                "no_speech_prob",
                1.0,
            )
        )

        compression_ratios.append(
            getattr(
                segment,
                "compression_ratio",
                999.0,
            )
        )

        words = getattr(
            segment,
            "words",
            None,
        )

        if words:
            for word in words:
                probability = getattr(
                    word,
                    "probability",
                    None,
                )

                if probability is None:
                    continue

                try:
                    probability = float(
                        probability
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if probability != probability:
                    continue

                total_words += 1

                word_probabilities.append(
                    probability
                )

                if (
                    probability
                    >= MIN_AVG_WORD_PROBABILITY
                ):
                    accepted_words += 1

    transcript = clean_text(
        " ".join(
            segment_texts
        )
    )

    average_logprob = safe_average(
        avg_logprobs,
        default=-10.0,
    )

    average_no_speech_probability = safe_average(
        no_speech_probs,
        default=1.0,
    )

    average_compression_ratio = safe_average(
        compression_ratios,
        default=999.0,
    )

    average_word_probability = safe_average(
        word_probabilities,
        default=0.0,
    )

    accepted_word_ratio = (
        accepted_words / total_words
        if total_words > 0
        else 0.0
    )

    repetition_ratio = calculate_repetition_ratio(
        transcript
    )

    language_probability = getattr(
        info,
        "language_probability",
        0.0,
    )

    duration = getattr(
        info,
        "duration",
        0.0,
    )

    try:
        language_probability = float(
            language_probability
        )
    except (
        TypeError,
        ValueError,
    ):
        language_probability = 0.0

    try:
        duration = float(
            duration
        )
    except (
        TypeError,
        ValueError,
    ):
        duration = 0.0

    metrics = {
        "average_logprob": round(
            average_logprob,
            4,
        ),
        "average_no_speech_probability": round(
            average_no_speech_probability,
            4,
        ),
        "average_compression_ratio": round(
            average_compression_ratio,
            4,
        ),
        "average_word_probability": round(
            average_word_probability,
            4,
        ),
        "accepted_word_ratio": round(
            accepted_word_ratio,
            4,
        ),
        "repetition_ratio": round(
            repetition_ratio,
            4,
        ),
        "language_probability": round(
            language_probability,
            4,
        ),
        "total_words": total_words,
        "accepted_words": accepted_words,
        "duration": round(
            duration,
            3,
        ),
    }

    return {
        "text": transcript,
        "segments": segments,
        "metrics": metrics,
    }


# ================================================================
# CONFIDENCE GATE
# ================================================================

def evaluate_transcription_confidence(
    transcription_data,
):
    """
    Evaluate transcription reliability.

    Uncertain speech is never passed to the security detector.
    """

    text = clean_text(
        transcription_data.get(
            "text",
            "",
        )
    )

    metrics = transcription_data.get(
        "metrics",
        {},
    )

    if not text:
        return {
            "status": "REJECT",
            "confidence": 0,
            "passed_checks": 0,
            "total_checks": 6,
            "reasons": [
                "No usable speech could be transcribed.",
            ],
            "checks": {},
        }

    average_logprob = float(
        metrics.get(
            "average_logprob",
            -10.0,
        )
    )

    average_no_speech_probability = float(
        metrics.get(
            "average_no_speech_probability",
            1.0,
        )
    )

    average_compression_ratio = float(
        metrics.get(
            "average_compression_ratio",
            999.0,
        )
    )

    average_word_probability = float(
        metrics.get(
            "average_word_probability",
            0.0,
        )
    )

    accepted_word_ratio = float(
        metrics.get(
            "accepted_word_ratio",
            0.0,
        )
    )

    repetition_ratio = float(
        metrics.get(
            "repetition_ratio",
            1.0,
        )
    )

    checks = {
        "avg_logprob": (
            average_logprob
            >= MIN_AVG_LOGPROB
        ),
        "no_speech": (
            average_no_speech_probability
            <= MAX_NO_SPEECH_PROB
        ),
        "word_probability": (
            average_word_probability
            >= MIN_AVG_WORD_PROBABILITY
        ),
        "accepted_word_ratio": (
            accepted_word_ratio
            >= MIN_ACCEPTED_WORD_RATIO
        ),
        "compression": (
            average_compression_ratio
            <= MAX_COMPRESSION_RATIO
        ),
        "repetition": (
            repetition_ratio
            <= MAX_REPETITION_RATIO
        ),
    }

    passed_checks = sum(
        1
        for passed in checks.values()
        if passed
    )

    reasons = []

    if not checks["avg_logprob"]:
        reasons.append(
            "Average Whisper log probability is below the reliability threshold."
        )

    if not checks["no_speech"]:
        reasons.append(
            "Whisper indicates a high probability of non-speech."
        )

    if not checks["word_probability"]:
        reasons.append(
            "Average word confidence is below the reliability threshold."
        )

    if not checks["accepted_word_ratio"]:
        reasons.append(
            "Too many transcribed words have low confidence."
        )

    if not checks["compression"]:
        reasons.append(
            "Whisper detected an unusually compressed or repetitive transcription pattern."
        )

    if not checks["repetition"]:
        reasons.append(
            "The transcription contains excessive repeated speech."
        )

    if passed_checks >= MIN_ACCEPTED_CHECKS:
        status = "ACCEPT"

    elif passed_checks >= MIN_REVIEW_CHECKS:
        status = "REVIEW"

    else:
        status = "REJECT"

    confidence = round(
        (
            passed_checks
            / len(checks)
        )
        * 100
    )

    if status == "REVIEW":
        reasons.insert(
            0,
            (
                "Speech transcription confidence is "
                f"insufficient for reliable security analysis "
                f"({confidence}%)."
            ),
        )

    elif status == "REJECT":
        reasons.insert(
            0,
            (
                "Speech transcription was rejected because "
                "the audio produced unreliable or repetitive text."
            ),
        )

    return {
        "status": status,
        "confidence": confidence,
        "passed_checks": passed_checks,
        "total_checks": len(checks),
        "checks": checks,
        "reasons": reasons,
    }


# ================================================================
# DETECTOR RESULT NORMALIZATION
# ================================================================

def normalize_detector_result(
    detector_result,
    source_text="",
):
    """
    Normalize the threat detector output.

    Supports multiple historical field names used by Aegis and
    restores security category metadata when a detector result
    contains the evidence but omits explicit categories.
    """

    if detector_result is None:
        detector_result = {}

    # ------------------------------------------------------------
    # Risk level
    # ------------------------------------------------------------

    risk = str(
        detector_result.get(
            "risk",
            detector_result.get(
                "risk_level",
                detector_result.get(
                    "level",
                    "LOW",
                ),
            ),
        )
        or "LOW"
    ).upper()

    # ------------------------------------------------------------
    # Score
    # ------------------------------------------------------------

    try:
        score = int(
            detector_result.get(
                "score",
                detector_result.get(
                    "risk_score",
                    0,
                ),
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        score = 0

    # ------------------------------------------------------------
    # Reasons
    # ------------------------------------------------------------

    reasons = detector_result.get(
        "reasons",
        detector_result.get(
            "indicators",
            detector_result.get(
                "detected_indicators",
                [],
            ),
        ),
    )

    if reasons is None:
        reasons = []

    if isinstance(
        reasons,
        str,
    ):
        reasons = [
            reasons
        ]

    reasons = [
        str(reason).strip()
        for reason in reasons
        if str(reason).strip()
    ]

    # ------------------------------------------------------------
    # URLs
    # ------------------------------------------------------------

    urls = detector_result.get(
        "urls",
        detector_result.get(
            "detected_urls",
            [],
        ),
    )

    if urls is None:
        urls = []

    if isinstance(
        urls,
        str,
    ):
        urls = [
            urls
        ]

    urls = [
        str(url).strip()
        for url in urls
        if str(url).strip()
    ]

    # ------------------------------------------------------------
    # Existing categories
    # ------------------------------------------------------------

    raw_categories = None

    category_keys = [
        "security_categories",
        "categories",
        "category",
        "indicator_categories",
        "threat_categories",
        "matched_categories",
    ]

    for key in category_keys:
        if key in detector_result:
            raw_categories = detector_result.get(
                key
            )

            if raw_categories:
                break

    if raw_categories is None:
        raw_categories = []

    if isinstance(
        raw_categories,
        str,
    ):
        raw_categories = [
            raw_categories
        ]

    categories = []

    for category in raw_categories:
        if isinstance(
            category,
            dict,
        ):
            category_value = (
                category.get("name")
                or category.get("category")
                or category.get("type")
            )
        else:
            category_value = category

        if category_value is None:
            continue

        category_value = str(
            category_value
        ).strip().lower()

        if (
            category_value
            and category_value not in categories
        ):
            categories.append(
                category_value
            )

    # ------------------------------------------------------------
    # Build evidence text from both the transcript and detector
    # reasons.
    # ------------------------------------------------------------

    evidence_text = clean_text(
        f"{source_text} {' '.join(reasons)}"
    ).lower()

    # ------------------------------------------------------------
    # Urgency
    # ------------------------------------------------------------

    urgency_patterns = [
        r"\burgent\b",
        r"\burgency\b",
        r"\bimmediately\b",
        r"\bact now\b",
        r"\bwithin\s+\d+\s+(?:hour|hours|minute|minutes)\b",
        r"\btoday\b",
        r"\blast warning\b",
        r"\bfinal notice\b",
        r"\bsuspended?\b",
        r"\bsuspension\b",
        r"\bblocked?\b",
        r"\blocked?\b",
        r"\brestricted?\b",
    ]

    if (
        any(
            re.search(
                pattern,
                evidence_text,
                re.IGNORECASE,
            )
            for pattern in urgency_patterns
        )
        and "urgency" not in categories
    ):
        categories.append(
            "urgency"
        )

    # ------------------------------------------------------------
    # Credentials
    # ------------------------------------------------------------

    credential_patterns = [
        r"\bpassword\b",
        r"\bpasscode\b",
        r"\botp\b",
        r"\bone[-\s]?time password\b",
        r"\br\s*o\s*t\s*p\b",
        r"\bpin\b",
        r"\bcvv\b",
        r"\bsecurity code\b",
        r"\bverification code\b",
        r"\bsensitive credential",
        r"\bsensitive credentials",
        r"\bfinancial information\b",
    ]

    if (
        any(
            re.search(
                pattern,
                evidence_text,
                re.IGNORECASE,
            )
            for pattern in credential_patterns
        )
        and "credentials" not in categories
    ):
        categories.append(
            "credentials"
        )

    # ------------------------------------------------------------
    # Payment
    # ------------------------------------------------------------

    payment_patterns = [
        r"\bpayment\b",
        r"\bprocessing fee\b",
        r"\bfee\b",
        r"\btransfer\b",
        r"\bpay\b",
        r"\bsend\s+(?:money|payment|amount)\b",
        r"\bmoney\b",
        r"\bamount\b",
        r"\bupi\b",
        r"\bcard\b",
    ]

    if (
        any(
            re.search(
                pattern,
                evidence_text,
                re.IGNORECASE,
            )
            for pattern in payment_patterns
        )
        and "payment" not in categories
    ):
        categories.append(
            "payment"
        )

    # ------------------------------------------------------------
    # Technical support
    # ------------------------------------------------------------

    technical_patterns = [
        r"\btechnical support\b",
        r"\btech support\b",
        r"\bremote access\b",
        r"\bremote control\b",
        r"\banydesk\b",
        r"\bteamviewer\b",
        r"\bsecurity technician\b",
        r"\bsupport scam\b",
    ]

    if (
        any(
            re.search(
                pattern,
                evidence_text,
                re.IGNORECASE,
            )
            for pattern in technical_patterns
        )
        and "technical_support" not in categories
    ):
        categories.append(
            "technical_support"
        )

    # ------------------------------------------------------------
    # Impersonation
    # ------------------------------------------------------------

    impersonation_patterns = [
        r"\bimpersonat(?:e|ing|ed)\b",
        r"\bbank security department\b",
        r"\bbank security\b",
        r"\bbank representative\b",
        r"\bofficial representative\b",
        r"\bcustomer support\b",
        r"\bcustomer care\b",
        r"\bsecurity department\b",
        r"\bfrom your bank\b",
        r"\bwe are your bank\b",
    ]

    if (
        any(
            re.search(
                pattern,
                evidence_text,
                re.IGNORECASE,
            )
            for pattern in impersonation_patterns
        )
        and "impersonation" not in categories
    ):
        categories.append(
            "impersonation"
        )

    # ------------------------------------------------------------
    # Link category
    # ------------------------------------------------------------

    if (
        urls
        and "link" not in categories
    ):
        categories.append(
            "link"
        )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    return {
        "risk_level": risk,
        "risk_score": score,
        "reasons": reasons,
        "urls": urls,
        "security_categories": categories,
    }


# ================================================================
# AUDIO SCAN
# ================================================================

def scan_audio(audio_path):
    """
    Full local audio security pipeline:

    Audio
      -> Faster-Whisper
      -> Confidence Gate
      -> Security Detector
    """

    transcription = transcribe_audio(
        audio_path
    )

    original_text = clean_text(
        transcription.get(
            "text",
            "",
        )
    )

    confidence_result = (
        evaluate_transcription_confidence(
            transcription
        )
    )

    status = confidence_result[
        "status"
    ]

    confidence = confidence_result[
        "confidence"
    ]

    confidence_reasons = (
        confidence_result[
            "reasons"
        ]
    )

    metrics = transcription.get(
        "metrics",
        {},
    )

    analysis_transcript = (
        normalize_security_transcript(
            original_text
        )
    )

    # ------------------------------------------------------------
    # Never analyze uncertain speech.
    # ------------------------------------------------------------

    if status != "ACCEPT":
        return {
            "risk_level": "LOW",
            "risk_score": 0,
            "risk": "LOW",
            "score": 0,
            "transcription_status": status,
            "transcription_confidence": confidence,
            "transcript": original_text,
            "analysis_transcript": analysis_transcript,
            "raw_transcript": original_text,
            "reasons": confidence_reasons,
            "urls": [],
            "security_categories": [],
            "transcription_metrics": metrics,
            "confidence_checks": confidence_result.get(
                "checks",
                {},
            ),
        }

    # ------------------------------------------------------------
    # Trusted speech goes to the security detector.
    # ------------------------------------------------------------

    detector_result = analyze_text(
        analysis_transcript
    )

    normalized_result = (
        normalize_detector_result(
            detector_result,
            source_text=analysis_transcript,
        )
    )

    return {
        "risk_level": normalized_result[
            "risk_level"
        ],
        "risk_score": normalized_result[
            "risk_score"
        ],
        "risk": normalized_result[
            "risk_level"
        ],
        "score": normalized_result[
            "risk_score"
        ],
        "transcription_status": "ACCEPT",
        "transcription_confidence": confidence,
        "transcript": original_text,
        "analysis_transcript": analysis_transcript,
        "raw_transcript": original_text,
        "reasons": normalized_result[
            "reasons"
        ],
        "urls": normalized_result[
            "urls"
        ],
        "security_categories": normalized_result[
            "security_categories"
        ],
        "transcription_metrics": metrics,
        "confidence_checks": confidence_result.get(
            "checks",
            {},
        ),
    }


# ================================================================
# UNTRUSTED SPEECH EXPLANATION
# ================================================================

def build_untrusted_explanation(
    result,
):
    status = str(
        result.get(
            "transcription_status",
            "REJECT",
        )
    ).upper()

    confidence = result.get(
        "transcription_confidence",
        0,
    )

    reasons = result.get(
        "reasons",
        [],
    )

    lines = [
        "WHY AEGIS DID NOT ANALYZE THIS SPEECH",
    ]

    if status == "REJECT":
        lines.append(
            "The speech transcription was rejected because the audio did not meet the reliability checks."
        )

    else:
        lines.append(
            "The speech transcription was placed under review because it did not meet the full reliability threshold."
        )

    lines.extend(
        [
            "",
            "TRANSCRIPTION RELIABILITY",
            f"Confidence gate result: {status}",
            f"Confidence estimate: {confidence}%",
        ]
    )

    if reasons:
        lines.extend(
            [
                "",
                "RELIABILITY WARNINGS",
            ]
        )

        for reason in reasons[:6]:
            lines.append(
                f"- {reason}"
            )

    lines.extend(
        [
            "",
            "SECURITY DECISION",
            "Aegis did not use the uncertain transcription to determine a security threat.",
        ]
    )

    return "\n".join(
        lines
    )


def build_no_speech_explanation():
    return (
        "No reliable speech transcription was available for security analysis. "
        "Aegis did not use the audio to determine a threat."
    )


# ================================================================
# LOCAL AI EXPLANATION
# ================================================================

def safe_explain_threat(
    result,
):
    transcript = clean_text(
        result.get(
            "analysis_transcript",
            result.get(
                "transcript",
                "",
            ),
        )
    )

    if not transcript:
        return build_no_speech_explanation()

    try:
        explanation = explain_threat(
            transcript,
            result,
        )

        if explanation is None:
            return ""

        explanation = str(
            explanation
        ).strip()

        # Prevent broken placeholder output such as:
        # "explanation"
        if (
            not explanation
            or explanation.lower()
            == "explanation"
        ):
            return ""

        return explanation

    except Exception as exc:
        return (
            "Aegis completed the trusted speech and security "
            f"analysis, but the local AI explanation was unavailable: "
            f"{exc}"
        )


# ================================================================
# COMPLETE AUDIO ANALYSIS WITH LOCAL AI
# ================================================================

def analyze_audio_with_ai(
    audio_path,
):
    """
    Complete local Aegis audio analysis.

    Returns:
        (result, explanation)
    """

    result = scan_audio(
        audio_path
    )

    status = str(
        result.get(
            "transcription_status",
            "REJECT",
        )
    ).upper()

    # ------------------------------------------------------------
    # Untrusted speech gets no security analysis.
    # ------------------------------------------------------------

    if status != "ACCEPT":
        explanation = (
            build_untrusted_explanation(
                result
            )
        )

        return (
            result,
            explanation,
        )

    # ------------------------------------------------------------
    # Trusted speech gets local Qwen explanation.
    # ------------------------------------------------------------

    explanation = safe_explain_threat(
        result
    )

    if not explanation:
        explanation = (
            "Trusted transcription was successfully analyzed by "
            "the Aegis security detector. No separate local AI "
            "explanation was available."
        )

    return (
        result,
        explanation,
    )