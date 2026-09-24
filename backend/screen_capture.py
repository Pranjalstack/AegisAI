import os
from datetime import datetime

import mss
from PIL import Image


def capture_screen():
    """Capture the primary monitor and save it as a PNG."""

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "screenshots"
    )

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        output_dir,
        f"screen_{timestamp}.png"
    )

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)

        image = Image.frombytes(
            "RGB",
            screenshot.size,
            screenshot.rgb
        )

        image.save(output_path)

    return output_path


if __name__ == "__main__":
    path = capture_screen()
    print(f"Screenshot saved to: {path}")