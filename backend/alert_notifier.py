import base64
import platform
import subprocess
import threading
import time
from typing import Iterable


ALERT_COOLDOWN_SECONDS = 15

_last_alert_time = 0.0
_alert_lock = threading.Lock()


def _clean_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if len(text) > 500:
        text = text[:497] + "..."

    return text


def _format_categories(categories: Iterable[object] | None) -> str:
    if not categories:
        return "security indicators"

    values = []

    for item in categories:
        value = _clean_text(item)

        if value and value not in values:
            values.append(value)

    if not values:
        return "security indicators"

    return ", ".join(values)


def _format_reasons(reasons: Iterable[object] | None) -> str:
    if not reasons:
        return ""

    values = []

    for item in reasons:
        value = _clean_text(item)

        if value and value not in values:
            values.append(value)

    if not values:
        return ""

    return "\n".join(f"• {value}" for value in values[:5])


def _build_message(
    risk_level: str,
    risk_score: int,
    transcript: str,
    categories: Iterable[object] | None,
    reasons: Iterable[object] | None,
) -> str:
    categories_text = _format_categories(categories)
    reasons_text = _format_reasons(reasons)

    message_parts = [
        "Aegis detected suspicious security-related activity.",
        "",
        f"Risk level: {risk_level}",
        f"Risk score: {risk_score}",
        f"Security categories: {categories_text}",
    ]

    if transcript:
        message_parts.extend(
            [
                "",
                "Detected speech:",
                transcript,
            ]
        )

    if reasons_text:
        message_parts.extend(
            [
                "",
                "Warning signs:",
                reasons_text,
            ]
        )

    message_parts.extend(
        [
            "",
            "Do not provide OTPs, passwords, PINs, CVVs, or banking details.",
            "Verify the request independently through an official channel.",
        ]
    )

    return "\n".join(message_parts)


def _launch_windows_alert(message: str) -> None:
    if platform.system().lower() != "windows":
        print("\n🚨 AEGIS SECURITY ALERT")
        print(message)
        return

    powershell_script = f"""
Add-Type -AssemblyName System.Windows.Forms
[System.Media.SystemSounds]::Exclamation.Play()

$wshell = New-Object -ComObject WScript.Shell

$message = @'
{message}
'@

$wshell.Popup(
    $message,
    8,
    'AEGIS SECURITY ALERT',
    48
)
"""

    encoded_command = base64.b64encode(
        powershell_script.encode("utf-16le")
    ).decode("ascii")

    try:
        subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-WindowStyle",
                "Hidden",
                "-EncodedCommand",
                encoded_command,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception as exc:
        print(f"\n[Notifier] Windows alert failed: {exc}")
        print("\n🚨 AEGIS SECURITY ALERT")
        print(message)


def notify_security_alert(
    risk_level: str,
    risk_score: int,
    transcript: str = "",
    categories: Iterable[object] | None = None,
    reasons: Iterable[object] | None = None,
) -> bool:
    global _last_alert_time

    now = time.monotonic()

    with _alert_lock:
        elapsed = now - _last_alert_time

        if elapsed < ALERT_COOLDOWN_SECONDS:
            return False

        _last_alert_time = now

    message = _build_message(
        risk_level=risk_level,
        risk_score=risk_score,
        transcript=transcript,
        categories=categories,
        reasons=reasons,
    )

    thread = threading.Thread(
        target=_launch_windows_alert,
        args=(message,),
        daemon=True,
    )

    thread.start()

    return True