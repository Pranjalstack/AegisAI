import pytesseract
from PIL import Image


# Tesseract is installed but not available in Windows PATH,
# so we specify its location directly.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text(image_path):
    """
    Extract visible text from an image using Tesseract OCR.
    """

    image = Image.open(image_path)

    text = pytesseract.image_to_string(image)

    return text.strip()


if __name__ == "__main__":
    image_path = r"screenshots\screen_20260918_123756.png"

    text = extract_text(image_path)

    print("\n--- OCR RESULT ---")
    print(text)