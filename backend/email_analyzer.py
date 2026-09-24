# backend/email_analyzer.py

import os

from backend.threat_detector import analyze_text
from backend.local_ai import explain_threat


def extract_email_text(file_path):
    """
    Read plain-text email or message content from a file.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.lower().endswith(".txt"):
        raise ValueError(
            "Only .txt email/message files are supported right now."
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read().strip()

    return text


def scan_email(file_path):
    """
    Read email/message text and analyze it for security threats.
    """

    text = extract_email_text(
        file_path
    )

    if not text:
        return {
            "text": "",
            "risk": "LOW",
            "score": 0,
            "reasons": [
                "No readable text found in the message"
            ],
            "urls": [],
        }

    result = analyze_text(
        text
    )

    return {
        "text": text,
        **result,
    }


def analyze_email_with_ai(file_path):
    """
    Perform complete email/message analysis using
    the rule engine and local AI reasoning.
    """

    result = scan_email(
        file_path
    )

    if not result["text"]:
        return {
            "scan": result,
            "explanation": (
                "No readable text was found in the message."
            ),
        }

    explanation = explain_threat(
        result["text"],
        result
    )

    return {
        "scan": result,
        "explanation": explanation,
    }


if __name__ == "__main__":

    email_path = (
        r"backend\test_email.txt"
    )

    result = analyze_email_with_ai(
        email_path
    )

    scan = result["scan"]

    print(
        "\n===== AEGIS EMAIL SECURITY SCAN ====="
    )

    print(
        f"Risk Level: {scan['risk']}"
    )

    print(
        f"Risk Score: {scan['score']}"
    )

    print(
        "\nDetected indicators:"
    )

    if scan["reasons"]:

        for reason in scan["reasons"]:

            print(
                f"• {reason}"
            )

    else:

        print(
            "• No major suspicious indicators detected"
        )

    if scan["urls"]:

        print(
            "\nLinks detected:"
        )

        for url in scan["urls"]:

            print(
                f"• {url}"
            )

    print(
        "\n===== AEGIS AI EMAIL EXPLANATION ====="
    )

    print(
        result["explanation"]
    )