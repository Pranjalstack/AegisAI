# backend/audio_analyzer.py

import torch

from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)

from backend.threat_detector import analyze_text
from backend.local_ai import explain_threat


MODEL_NAME = "openai/whisper-base"


print(
    "Loading Aegis local speech model...",
    flush=True,
)


processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


dtype = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)


model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
)


device = (
    0
    if torch.cuda.is_available()
    else -1
)


speech_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
)


print(
    "Aegis local speech model loaded.",
    flush=True,
)


def transcribe_audio(audio_path):
    """
    Transcribe an audio file using local Whisper.
    """

    result = speech_pipeline(
        audio_path,
        generate_kwargs={
            "task": "transcribe",
        },
    )

    return result["text"].strip()


def scan_audio(audio_path):
    """
    Transcribe audio and analyze the transcript
    for security threats.
    """

    text = transcribe_audio(
        audio_path
    )

    if not text:

        return {
            "text": "",
            "risk": "LOW",
            "score": 0,
            "reasons": [
                "No speech could be transcribed"
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


def analyze_audio_with_ai(audio_path):
    """
    Perform complete audio security analysis using
    Whisper, the rule engine, and local Qwen reasoning.
    """

    result = scan_audio(
        audio_path
    )

    if not result["text"]:

        return {
            "scan": result,
            "explanation": (
                "No speech could be transcribed "
                "from the audio."
            ),
        }

    explanation = explain_threat(
        result["text"],
        result,
    )

    return {
        "scan": result,
        "explanation": explanation,
    }


if __name__ == "__main__":

    audio_path = (
        r"backend\test_call.wav"
    )

    result = analyze_audio_with_ai(
        audio_path
    )

    scan = result["scan"]


    print(
        "\n===== AEGIS AUDIO SECURITY SCAN ====="
    )

    print(
        f"Risk Level: {scan['risk']}"
    )

    print(
        f"Risk Score: {scan['score']}"
    )


    print(
        "\nTranscript:"
    )

    print(
        scan["text"]
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
        "\n===== AEGIS AI AUDIO EXPLANATION ====="
    )

    print(
        result["explanation"]
    )