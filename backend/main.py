import time

from screen_capture import capture_screen
from analyzer import extract_text
from threat_detector import analyze_text
from local_ai import explain_threat, analyze_image


def run_screen_analysis():
    print("\n===== AEGIS AI SCREEN ANALYSIS =====")

    # 1. Capture screen
    print("\n[1/5] Preparing to capture screen...")

    for seconds in range(5, 0, -1):
        print(f"Capturing in {seconds}...", end="\r")
        time.sleep(1)

    print("\nCapturing screen...")

    image_path = capture_screen()
    print(f"Screenshot saved: {image_path}")

    # 2. Extract text with OCR
    print("\n[2/5] Extracting text with OCR...")

    text = extract_text(image_path)

    if not text:
        print("No readable text detected on screen.")
        text = ""
    else:
        print("OCR completed.")

    # 3. Rule-based threat detection
    print("\n[3/5] Analyzing text for security risks...")

    result = analyze_text(text)

    print("\n===== AEGIS TEXT RESULT =====")
    print(f"Risk Level: {result['risk']}")
    print(f"Risk Score: {result['score']}")

    print("\nDetected indicators:")

    if result["reasons"]:
        for reason in result["reasons"]:
            print(f"• {reason}")
    else:
        print("• No major suspicious indicators detected")

    if result["urls"]:
        print("\nLinks detected:")

        for url in result["urls"]:
            print(f"• {url}")

    # 4. Visual screenshot analysis
    print("\n[4/5] Analyzing screenshot visually...")

    vision_result = analyze_image(image_path)

    print("\n===== AEGIS VISION RESULT =====")
    print(vision_result)

    # 5. Local AI reasoning
    print("\n[5/5] Generating local AI explanation...")

    explanation = explain_threat(text, result)

    print("\n===== AEGIS AI EXPLANATION =====")
    print(explanation)


if __name__ == "__main__":
    run_screen_analysis()