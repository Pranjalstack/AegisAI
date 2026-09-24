# backend/api.py

import threading
import time
import tempfile
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.screen_capture import capture_screen
from backend.aegis_engine import (
    analyze_screen,
    analyze_file,
    analyze_message,
    analyze_audio,
)


app = FastAPI(title="Aegis AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


scan_lock = threading.Lock()

MAX_FILE_SIZE = 15 * 1024 * 1024

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".webm",
}


@app.get("/")
def root():
    return {
        "name": "Aegis AI",
        "status": "online",
    }


@app.post("/scan-screen")
def scan_screen():

    if not scan_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={
                "error": True,
                "message": (
                    "Aegis is already analyzing something. "
                    "Please wait for the current scan to finish."
                ),
            },
        )

    try:

        print(
            "\n===== NEW SCREEN SCAN REQUEST =====",
            flush=True,
        )

        print(
            "[1/5] Capture begins in 5 seconds. "
            "Switch to the screen you want Aegis to inspect.",
            flush=True,
        )

        for seconds in range(5, 0, -1):

            print(
                f"Capturing in {seconds}...",
                flush=True,
            )

            time.sleep(1)

        print(
            "[1/5] Capturing screen...",
            flush=True,
        )

        image_path = capture_screen()

        print(
            f"[1/5] Screenshot captured: {image_path}",
            flush=True,
        )

        print(
            "[2-5/5] Running unified screen analysis...",
            flush=True,
        )

        result = analyze_screen(
            image_path
        )

        print(
            "[2-5/5] Unified screen analysis complete.",
            flush=True,
        )

        print(
            "===== SCREEN SCAN COMPLETE =====",
            flush=True,
        )

        return result

    except Exception as error:

        print(
            f"SCREEN SCAN ERROR: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": str(error),
            },
        )

    finally:

        scan_lock.release()


@app.post("/scan-file")
async def scan_file(
    request: Request,
    filename: str = "document.pdf",
):

    if not scan_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={
                "error": True,
                "message": (
                    "Aegis is already analyzing something. "
                    "Please wait for the current scan to finish."
                ),
            },
        )

    temp_path = None

    try:

        print(
            "\n===== NEW FILE SCAN REQUEST =====",
            flush=True,
        )

        safe_name = Path(
            filename
        ).name

        if not safe_name.lower().endswith(
            ".pdf"
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "message": (
                        "Only PDF files are supported right now."
                    ),
                },
            )

        file_data = await request.body()

        if not file_data:

            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "message": (
                        "The selected file is empty."
                    ),
                },
            )

        if len(file_data) > MAX_FILE_SIZE:

            return JSONResponse(
                status_code=413,
                content={
                    "error": True,
                    "message": (
                        "The PDF is larger than the 15 MB limit."
                    ),
                },
            )

        print(
            f"[1/2] Receiving PDF: {safe_name}",
            flush=True,
        )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pdf",
            delete=False,
        ) as temp_file:

            temp_file.write(
                file_data
            )

            temp_path = temp_file.name

        print(
            "[2/2] Running unified file analysis...",
            flush=True,
        )

        result = await run_in_threadpool(
            analyze_file,
            temp_path,
            safe_name,
        )

        print(
            "===== FILE SCAN COMPLETE =====",
            flush=True,
        )

        return result

    except Exception as error:

        print(
            f"FILE SCAN ERROR: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": str(error),
            },
        )

    finally:

        if temp_path:

            try:
                Path(
                    temp_path
                ).unlink(
                    missing_ok=True
                )

            except Exception:
                pass

        scan_lock.release()


@app.post("/scan-message")
async def scan_message(
    request: Request
):

    if not scan_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={
                "error": True,
                "message": (
                    "Aegis is already analyzing something. "
                    "Please wait for the current scan to finish."
                ),
            },
        )

    try:

        print(
            "\n===== NEW MESSAGE SCAN REQUEST =====",
            flush=True,
        )

        payload = await request.json()

        message = str(
            payload.get(
                "message",
                "",
            )
        ).strip()

        if not message:

            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "message": (
                        "Please enter a message to analyze."
                    ),
                },
            )

        if len(message) > 12000:

            return JSONResponse(
                status_code=413,
                content={
                    "error": True,
                    "message": (
                        "The message is too long. "
                        "Please keep it under 12,000 characters."
                    ),
                },
            )

        print(
            f"[1/1] Running unified message analysis "
            f"on {len(message)} characters...",
            flush=True,
        )

        result = await run_in_threadpool(
            analyze_message,
            message,
        )

        print(
            "===== MESSAGE SCAN COMPLETE =====",
            flush=True,
        )

        return result

    except Exception as error:

        print(
            f"MESSAGE SCAN ERROR: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": str(error),
            },
        )

    finally:

        scan_lock.release()


@app.post("/scan-audio")
async def scan_audio(
    request: Request,
    filename: str = "audio.wav",
):

    if not scan_lock.acquire(blocking=False):
        return JSONResponse(
            status_code=429,
            content={
                "error": True,
                "message": (
                    "Aegis is already analyzing something. "
                    "Please wait for the current scan to finish."
                ),
            },
        )

    temp_path = None

    try:

        print(
            "\n===== NEW AUDIO SCAN REQUEST =====",
            flush=True,
        )

        safe_name = Path(
            filename
        ).name

        extension = Path(
            safe_name
        ).suffix.lower()

        if extension not in ALLOWED_AUDIO_EXTENSIONS:

            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "message": (
                        "Unsupported audio format. "
                        "Use WAV, MP3, M4A, OGG, or WEBM."
                    ),
                },
            )

        audio_data = await request.body()

        if not audio_data:

            return JSONResponse(
                status_code=400,
                content={
                    "error": True,
                    "message": (
                        "The selected audio file is empty."
                    ),
                },
            )

        if len(audio_data) > MAX_FILE_SIZE:

            return JSONResponse(
                status_code=413,
                content={
                    "error": True,
                    "message": (
                        "The audio file is larger than the 15 MB limit."
                    ),
                },
            )

        print(
            f"[1/2] Receiving audio: {safe_name}",
            flush=True,
        )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=extension,
            delete=False,
        ) as temp_file:

            temp_file.write(
                audio_data
            )

            temp_path = temp_file.name

        print(
            "[2/2] Running unified audio analysis...",
            flush=True,
        )

        result = await run_in_threadpool(
            analyze_audio,
            temp_path,
            safe_name,
        )

        print(
            "===== AUDIO SCAN COMPLETE =====",
            flush=True,
        )

        return result

    except Exception as error:

        print(
            f"AUDIO SCAN ERROR: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": str(error),
            },
        )

    finally:

        if temp_path:

            try:
                Path(
                    temp_path
                ).unlink(
                    missing_ok=True
                )

            except Exception:
                pass

        scan_lock.release()