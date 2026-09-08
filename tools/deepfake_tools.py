"""Tools for preliminary image-based Deepfake analysis."""

import os

# OpenCV is used to read the image and perform basic computer-vision measurements such as grayscale conversion, sharpness, and edge detection.
import cv2

# NumPy is used to calculate numerical image measurements such as average brightness, edge density, and image contrast.
import numpy as np


def media_indicator_analyzer(
    image_path: str,
) -> dict:
    """
    Analyse an image for visual manipulation indicators; accepts an image
    path and returns image measurements, indicators, and a preliminary risk score.
    """

    # Validate that an image path was provided before attempting to access the image file.
    if not image_path:
        raise ValueError(
            "Image path was not provided."
        )

    # Check whether the specified image file exists.
    # This prevents the tool from attempting to process an invalid path.
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    # Extract the file extension so the tool can verify that the submitted image uses a supported format.
    extension = os.path.splitext(
        image_path
    )[1].lower()

    # Define the image formats supported by the Deepfake analysis tool.
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    # Reject unsupported file formats before processing the image.
    if extension not in supported_extensions:
        raise ValueError(
            "Unsupported image format. "
            "Use JPG, JPEG, PNG, BMP, or WEBP."
        )

    # Load the image using OpenCV.
    image = cv2.imread(image_path)

    # If OpenCV cannot read the image, raise an error so the Deepfake Agent can handle the failure through its fallback.
    if image is None:
        raise ValueError(
            "Unable to read the image file."
        )

    # Store all visual indicators detected during the analysis.
    indicators = []

    # --------------------------------------------------------
    # Convert image to grayscale
    # --------------------------------------------------------

    # Convert the image from BGR colour format to grayscale.
    # Grayscale simplifies the numerical analysis of brightness, sharpness, edges, and contrast.
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------------
    # Brightness analysis
    # --------------------------------------------------------

    # Calculate the average brightness of all pixels in the image.
    brightness = float(
        np.mean(gray)
    )

    # Extremely dark or extremely bright images may provide unusual visual characteristics for further inspection.
    if brightness < 35 or brightness > 225:
        indicators.append(
            "Unusual image brightness"
        )

    # --------------------------------------------------------
    # Sharpness analysis
    # --------------------------------------------------------

    # Calculate image sharpness using the variance of the Laplacian. Lower values generally indicate less detail or a blurrier image.
    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    # Flag images with low measured sharpness as a preliminary visual indicator that may require additional verification.
    if sharpness < 100:
        indicators.append(
            "Low visual sharpness"
        )

    # --------------------------------------------------------
    # Edge analysis
    # --------------------------------------------------------

    # Detect edges in the grayscale image using the Canny edge-detection algorithm.
    edges = cv2.Canny(
        gray,
        100,
        200,
    )

    # Calculate the proportion of pixels classified as edges.
    # This provides a basic measurement of the amount of visible detail.
    edge_density = float(
        np.mean(edges > 0)
    )

    # Flag images with very low edge density as having low visible edge detail.
    if edge_density < 0.03:
        indicators.append(
            "Low edge detail"
        )

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    # Obtain the image height and width from the grayscale image.
    height, width = gray.shape

    # Very small images provide less visual information and may therefore provide less reliable evidence for analysis.
    if width < 500 or height < 500:
        indicators.append(
            "Low image resolution"
        )

    # --------------------------------------------------------
    # Contrast analysis
    # --------------------------------------------------------

    # Calculate the standard deviation of pixel intensity values.
    # This provides a basic measurement of image contrast.
    contrast = float(
        np.std(gray)
    )

    # Flag images with low measured contrast as a preliminary visual indicator requiring additional authenticity checks.
    if contrast < 35:
        indicators.append(
            "Unusual image contrast"
        )

    # --------------------------------------------------------
    # Preliminary risk score
    # --------------------------------------------------------

    # Count the total number of visual indicators detected.
    indicator_count = len(indicators)

    # Assign a high preliminary risk score when three or more visual indicators are detected.
    if indicator_count >= 3:
        risk_score = 75
        risk_level = "High"

    # Assign a medium preliminary risk score when exactly two visual indicators are detected.
    elif indicator_count == 2:
        risk_score = 55
        risk_level = "Medium"

    # Assign a lower medium preliminary score when one indicator is detected.
    elif indicator_count == 1:
        risk_score = 35
        risk_level = "Medium"

    # Assign the lowest preliminary score when no indicators are detected. This does not prove image authenticity.
    else:
        risk_score = 15
        risk_level = "Low"

    # Return the complete technical analysis so the Deepfake Agent can use the measurements and indicators in its assessment.
    return {
        "media_type": "image",
        "image_path": image_path,
        "width": width,
        "height": height,
        "brightness": round(
            brightness,
            2,
        ),
        "sharpness": round(
            sharpness,
            2,
        ),
        "edge_density": round(
            edge_density,
            4,
        ),
        "contrast": round(
            contrast,
            2,
        ),
        "indicators": indicators,
        "indicator_count": indicator_count,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }