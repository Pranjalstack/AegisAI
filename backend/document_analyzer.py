# backend/document_analyzer.py

import os
import pymupdf
import pytesseract

from PIL import Image

from backend.threat_detector import analyze_text
from backend.local_ai import explain_threat


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_pdf_text(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.lower().endswith(".pdf"):
        raise ValueError(
            "Only PDF files are supported right now."
        )

    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        text = page.get_text()

        if text.strip():
            pages.append(
                f"--- Page {page_number} ---\n"
                f"{text.strip()}"
            )

    document.close()

    return "\n\n".join(pages).strip()


def extract_pdf_text_with_ocr(file_path):
    document = pymupdf.open(file_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2)
        )

        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )

        text = pytesseract.image_to_string(
            image
        ).strip()

        if text:
            pages.append(
                f"--- Page {page_number} ---\n"
                f"{text}"
            )

    document.close()

    return "\n\n".join(pages).strip()


def extract_pdf_text_auto(file_path):
    text = extract_pdf_text(file_path)

    if text:
        return text

    print(
        "No selectable PDF text found.",
        flush=True,
    )

    print(
        "Falling back to OCR...",
        flush=True,
    )

    return extract_pdf_text_with_ocr(file_path)


def scan_pdf(file_path):
    text = extract_pdf_text_auto(
        file_path
    )

    if not text:
        return {
            "text": "",
            "risk": "LOW",
            "score": 0,
            "reasons": [
                "No readable text detected in the PDF"
            ],
            "urls": [],
        }

    result = analyze_text(text)

    return {
        "text": text,
        **result,
    }


def analyze_pdf_with_ai(file_path):
    result = scan_pdf(file_path)

    if not result["text"]:
        return {
            "scan": result,
            "explanation": (
                "No readable text was found in the PDF."
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