"""OCR tool for extracting text from an image."""

# OpenCV is used to read the submitted image and perform basic image preprocessing before OCR.
import cv2

# Pytesseract connects the Python program to the Tesseract OCR engine so text can be extracted from the image.
import pytesseract


# Tesseract OCR executable

# This specifies the location of the Tesseract program installed on the local computer so pytesseract can perform OCR.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text_from_image(
    image_path: str,
) -> str:
    """
    Extract text from an image; accepts an image path and
    returns the detected text.
    """

    # Human developer modification:
    # Disable OpenCV warning messages so invalid image paths
    # produce a clean user-facing error instead of technical logs.
    cv2.utils.logging.setLogLevel(
        cv2.utils.logging.LOG_LEVEL_SILENT
    )

    # Read the submitted image from the provided file path.
    image = cv2.imread(
        image_path
    )

    # Stop the OCR process if the image cannot be loaded.
    # This prevents invalid image data from being passed to the preprocessing and OCR stages.
    if image is None:
        raise ValueError(
            "Unable to read the image file."
        )

    # Convert the image from colour format to grayscale.
    # Grayscale simplifies the image and makes text processing easier for the OCR engine.
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Improve text visibility by converting the grayscale image into a binary black-and-white image using Otsu thresholding.
    # This helps separate text from the image background.
    gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )[1]

    # Send the preprocessed image to Tesseract OCR to detect and extract any visible text.
    text = pytesseract.image_to_string(
        gray
    )

    # Remove unnecessary whitespace from the OCR result and return the cleaned text to the workflow.
    return text.strip()