# backend/aegis_engine.py

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class AegisResult:
    """
    Standard result returned by every Aegis analysis mode.
    """

    type: str

    risk: str

    score: int

    reasons: list[str] = field(
        default_factory=list
    )

    urls: list[str] = field(
        default_factory=list
    )

    explanation: str = ""

    transcript: str = ""

    vision: str = ""

    filename: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self):
        """
        Convert the result into a JSON-compatible dictionary.
        """

        return asdict(self)


def create_result(
    scan_type: str,
    risk: str = "LOW",
    score: int = 0,
    reasons=None,
    urls=None,
    explanation: str = "",
    transcript: str = "",
    vision: str = "",
    filename: str = "",
    metadata=None,
):
    """
    Create a standardized Aegis result.
    """

    return AegisResult(
        type=str(scan_type),

        risk=str(risk).upper(),

        score=int(score),

        reasons=list(
            reasons or []
        ),

        urls=list(
            urls or []
        ),

        explanation=str(
            explanation or ""
        ),

        transcript=str(
            transcript or ""
        ),

        vision=str(
            vision or ""
        ),

        filename=str(
            filename or ""
        ),

        metadata=dict(
            metadata or {}
        ),
    )


def result_from_detector(
    scan_type: str,
    detector_result: dict,
    explanation: str = "",
    transcript: str = "",
    vision: str = "",
    filename: str = "",
    metadata=None,
):
    """
    Convert the existing threat detector result into
    the unified Aegis result format.
    """

    return create_result(
        scan_type=scan_type,

        risk=detector_result.get(
            "risk",
            "LOW"
        ),

        score=detector_result.get(
            "score",
            0
        ),

        reasons=detector_result.get(
            "reasons",
            []
        ),

        urls=detector_result.get(
            "urls",
            []
        ),

        explanation=explanation,

        transcript=transcript,

        vision=vision,

        filename=filename,

        metadata=metadata,
    )


def clean_result(result):
    """
    Normalize an Aegis result before returning it to the UI.
    """

    if isinstance(
        result,
        AegisResult
    ):
        result = result.to_dict()

    if not isinstance(
        result,
        dict
    ):
        raise TypeError(
            "Aegis result must be a dictionary."
        )

    risk = str(
        result.get(
            "risk",
            "LOW"
        )
    ).upper()

    if risk not in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }:
        risk = "LOW"

    try:
        score = int(
            result.get(
                "score",
                0
            )
        )
    except (
        TypeError,
        ValueError
    ):
        score = 0

    score = max(
        0,
        score
    )

    reasons = result.get(
        "reasons",
        []
    )

    if not isinstance(
        reasons,
        list
    ):
        reasons = [
            str(reasons)
        ]

    urls = result.get(
        "urls",
        []
    )

    if not isinstance(
        urls,
        list
    ):
        urls = [
            str(urls)
        ]

    metadata = result.get(
        "metadata",
        {}
    )

    if not isinstance(
        metadata,
        dict
    ):
        metadata = {}

    return {
        "type": str(
            result.get(
                "type",
                "unknown"
            )
        ),

        "risk": risk,

        "score": score,

        "reasons": [
            str(reason)
            for reason in reasons
        ],

        "urls": [
            str(url)
            for url in urls
        ],

        "explanation": str(
            result.get(
                "explanation",
                ""
            )
        ),

        "transcript": str(
            result.get(
                "transcript",
                ""
            )
        ),

        "vision": str(
            result.get(
                "vision",
                ""
            )
        ),

        "filename": str(
            result.get(
                "filename",
                ""
            )
        ),

        "metadata": metadata,
    }


def analyze_message(
    message: str
):
    """
    Analyze a text message through the unified engine.
    """

    message = str(
        message or ""
    ).strip()

    if not message:
        return clean_result(
            create_result(
                scan_type="message",
                risk="LOW",
                score=0,
                reasons=[
                    "No message content was provided."
                ],
            )
        )

    from backend.threat_detector import analyze_text
    from backend.local_ai import explain_threat

    detector_result = analyze_text(
        message
    )

    explanation = explain_threat(
        message,
        detector_result
    )

    result = result_from_detector(
        scan_type="message",
        detector_result=detector_result,
        explanation=explanation,
    )

    return clean_result(
        result
    )


def analyze_file(
    file_path: str,
    filename: str = ""
):
    """
    Analyze a PDF through the unified engine.
    Supports normal text PDFs and scanned PDFs.
    """

    from backend.document_analyzer import (
        analyze_pdf_with_ai
    )

    result = analyze_pdf_with_ai(
        file_path
    )

    scan = result.get(
        "scan",
        {}
    )

    unified = result_from_detector(
        scan_type="file",
        detector_result=scan,
        explanation=result.get(
            "explanation",
            ""
        ),
        filename=filename,
    )

    return clean_result(
        unified
    )


def analyze_audio(
    audio_path: str,
    filename: str = ""
):
    """
    Analyze an audio recording through the unified engine.
    """

    from backend.audio_analyzer import (
        analyze_audio_with_ai
    )

    result = analyze_audio_with_ai(
        audio_path
    )

    scan = result.get(
        "scan",
        {}
    )

    unified = result_from_detector(
        scan_type="audio",
        detector_result=scan,
        explanation=result.get(
            "explanation",
            ""
        ),
        transcript=scan.get(
            "text",
            ""
        ),
        filename=filename,
    )

    return clean_result(
        unified
    )


def analyze_screen(
    image_path: str
):
    """
    Analyze a captured screen through the unified engine.

    The actual capture happens outside this function so the
    engine remains independent from the UI/API layer.
    """

    from backend.analyzer import extract_text
    from backend.threat_detector import analyze_text
    from backend.local_ai import (
        explain_threat,
        analyze_image,
    )

    text = extract_text(
        image_path
    )

    if not text:
        text = ""

    detector_result = analyze_text(
        text
    )

    vision_result = analyze_image(
        image_path
    )

    explanation = explain_threat(
        text,
        detector_result
    )

    unified = result_from_detector(
        scan_type="screen",
        detector_result=detector_result,
        explanation=explanation,
        vision=vision_result,
    )

    return clean_result(
        unified
    )