# backend/api.py

import tempfile
import threading
import time
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

from backend.live_audio import (
    start_live_guard,
    stop_live_guard,
    get_live_status,
    try_start_operation,
    finish_operation,
)


# ================================================================
# APP
# ================================================================

app = FastAPI(
    title="Aegis AI"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# CONFIGURATION
# ================================================================

MAX_FILE_SIZE = (
    15 * 1024 * 1024
)

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".webm",
}


# ================================================================
# HELPERS
# ================================================================

def operation_busy_response():
    status = get_live_status()

    operation = status.get(
        "operation"
    )

    if operation:
        message = (
            "Aegis is currently busy with "
            f"{operation}. "
            "Please wait for the current operation to finish."
        )
    else:
        message = (
            "Aegis is already analyzing something. "
            "Please wait for the current scan to finish."
        )

    return JSONResponse(
        status_code=429,
        content={
            "error": True,
            "message": message,
        },
    )


def begin_operation(
    operation_name
):
    acquired, existing = (
        try_start_operation(
            operation_name
        )
    )

    if not acquired:
        return False, JSONResponse(
            status_code=429,
            content={
                "error": True,
                "message": (
                    "Aegis is currently busy with "
                    f"{existing}. "
                    "Please wait for the current operation to finish."
                ),
            },
        )

    return True, None


# ================================================================
# ROOT
# ================================================================

@app.get("/")
def root():
    return {
        "name": "Aegis AI",
        "status": "online",
        "live_audio": get_live_status(),
    }


# ================================================================
# LIVE AUDIO STATUS
# ================================================================

@app.get("/live-audio/status")
def live_audio_status():
    return {
        "error": False,
        **get_live_status(),
    }


# ================================================================
# LIVE AUDIO START
# ================================================================

@app.post("/live-audio/start")
def live_audio_start():
    acquired, error_response = (
        begin_operation(
            "live-audio"
        )
    )

    if not acquired:
        return error_response

    # Release the temporary reservation. The actual live guard
    # immediately performs its own operation reservation.
    finish_operation(
        "live-audio"
    )

    result = start_live_guard()

    if not result.get(
        "started",
        False,
    ):
        return JSONResponse(
            status_code=409,
            content={
                "error": True,
                "message": result.get(
                    "message",
                    "Unable to start Live Audio Guard.",
                ),
            },
        )

    return {
        "error": False,
        **result,
        "status": get_live_status(),
    }


# ================================================================
# LIVE AUDIO STOP
# ================================================================

@app.post("/live-audio/stop")
def live_audio_stop():
    result = stop_live_guard()

    return {
        "error": False,
        **result,
        "status": get_live_status(),
    }


# ================================================================
# SCREEN SCAN
# ================================================================

@app.post("/scan-screen")
def scan_screen():
    acquired, error_response = (
        begin_operation(
            "screen-scan"
        )
    )

    if not acquired:
        return error_response

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

        for seconds in range(
            5,
            0,
            -1,
        ):
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
            "SCREEN SCAN ERROR: "
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
        finish_operation(
            "screen-scan"
        )


# ================================================================
# FILE SCAN
# ================================================================

@app.post("/scan-file")
async def scan_file(
    request: Request,
    filename: str = "document.pdf",
):
    acquired, error_response = (
        begin_operation(
            "file-scan"
        )
    )

    if not acquired:
        return error_response

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
            "FILE SCAN ERROR: "
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

        finish_operation(
            "file-scan"
        )


# ================================================================
# MESSAGE SCAN
# ================================================================

@app.post("/scan-message")
async def scan_message(
    request: Request
):
    acquired, error_response = (
        begin_operation(
            "message-scan"
        )
    )

    if not acquired:
        return error_response

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
            "[1/1] Running unified message analysis "
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
            "MESSAGE SCAN ERROR: "
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
        finish_operation(
            "message-scan"
        )


# ================================================================
# MANUAL AUDIO SCAN
# ================================================================

@app.post("/scan-audio")
async def scan_audio(
    request: Request,
    filename: str = "audio.wav",
):
    acquired, error_response = (
        begin_operation(
            "audio-scan"
        )
    )

    if not acquired:
        return error_response

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

        if (
            extension
            not in ALLOWED_AUDIO_EXTENSIONS
        ):
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
            "AUDIO SCAN ERROR: "
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

        finish_operation(
            "audio-scan"
        )