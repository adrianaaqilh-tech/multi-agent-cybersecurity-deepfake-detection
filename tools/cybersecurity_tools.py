"""Tools for preliminary cybersecurity analysis."""

import re


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs from image text.

    Args:
        text: Text extracted from the submitted image.

    Returns:
        A list of detected URLs.
    """

    # Human developer modification:
    # Return an empty list when no text is available because
    # there is nothing to scan for URLs.
    if not text:
        return []

    # Human developer modification:
    # Detect URLs beginning with http, https, or www.
    pattern = r"https?://[^\s]+|www\.[^\s]+"

    return re.findall(
        pattern,
        text,
        re.IGNORECASE,
    )


def url_threat_checker(url: str) -> dict:
    """
    Check a URL for basic suspicious patterns.

    Args:
        url: A single URL detected from the image text.

    Returns:
        A dictionary containing the URL, detected indicators,
        indicator count, and risk level.
    """

    # Human developer modification:
    # Store suspicious URL indicators detected during the scan.
    indicators = []

    # Human developer modification:
    # These patterns represent common suspicious URL characteristics
    # such as fake login pages, verification requests, urgency,
    # and account-related wording.
    suspicious_patterns = [
        (r"@", "URL contains @ symbol"),
        (r"-login", "Suspicious login URL pattern"),
        (r"-verify", "Suspicious verification URL pattern"),
        (r"urgent", "Urgency-related URL wording"),
        (r"secure-update", "Suspicious security update wording"),
        (r"free", "Potentially suspicious free offer wording"),
        (r"account", "Account-related URL"),
        (r"confirm", "Account confirmation wording"),
        (r"verify", "Verification wording"),
        (r"password", "Password-related URL wording"),
        (r"bank", "Bank-related URL wording"),
    ]

    # Check the URL against every suspicious pattern.
    for pattern, description in suspicious_patterns:
        if re.search(
            pattern,
            url,
            re.IGNORECASE,
        ):
            indicators.append(description)

    # Assign a risk level based on the number of URL indicators.
    if len(indicators) >= 3:
        risk_level = "High"

    elif len(indicators) >= 1:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    return {
        "url": url,
        "indicators": indicators,
        "indicator_count": len(indicators),
        "risk_level": risk_level,
    }


def cybersecurity_indicator_checker(
    text: str,
) -> dict:
    """
    Check image text for cybersecurity indicators.

    Args:
        text: Text extracted from the submitted image.

    Returns:
        A dictionary containing detected URLs, indicators,
        indicator count, risk level, and URL analysis.
    """

    # Human developer modification:
    # If OCR returns no text, return a safe low-risk result
    # instead of attempting to analyse empty content.
    if not text:
        return {
            "urls": [],
            "indicators": [],
            "indicator_count": 0,
            "risk_level": "Low",
            "url_analysis": [],
        }

    # Store cybersecurity indicators detected from OCR text.
    indicators = []

    # Human developer modification:
    # Include common phishing, scam, fraud, malware, social
    # engineering, banking, and suspicious-message keywords.
    suspicious_keywords = [
        (
            "phishing",
            "Phishing-related content",
        ),
        (
            "phish",
            "Possible phishing-related content",
        ),
        (
            "scam",
            "Scam-related content",
        ),
        (
            "fraud",
            "Fraud-related content",
        ),
        (
            "malware",
            "Malware-related content",
        ),
        (
            "ransomware",
            "Ransomware-related content",
        ),
        (
            "hack",
            "Hacking-related content",
        ),
        (
            "hacking",
            "Hacking-related content",
        ),
        (
            "click",
            "Call-to-action requesting the user to click",
        ),
        (
            "verify",
            "Verification request",
        ),
        (
            "urgent",
            "Urgency language",
        ),
        (
            "suspended",
            "Account suspension warning",
        ),
        (
            "password",
            "Password-related request",
        ),
        (
            "login",
            "Login-related request",
        ),
        (
            "claim",
            "Potential claim/reward bait",
        ),
        (
            "free",
            "Potentially suspicious free offer",
        ),
        (
            "prize",
            "Prize-related bait",
        ),
        (
            "investment",
            "Investment-related content",
        ),
        (
            "bank",
            "Banking-related content",
        ),
        (
            "security",
            "Security-related content",
        ),
    ]

    # Scan the extracted image text for suspicious keywords.
    lower_text = text.lower()

    for keyword, description in suspicious_keywords:
        if keyword in lower_text:
            indicators.append(description)

    # Extract URLs from the OCR text.
    urls = extract_urls(text)

    # Analyse every detected URL independently.
    url_results = []

    for url in urls:
        url_results.append(
            url_threat_checker(url)
        )

    # Add URL indicators to the overall cybersecurity findings.
    for result in url_results:
        indicators.extend(
            result["indicators"]
        )

    # Calculate the preliminary overall risk level.
    if len(indicators) >= 3:
        risk_level = "High"

    elif len(indicators) >= 1:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    return {
        "urls": urls,
        "indicators": indicators,
        "indicator_count": len(indicators),
        "risk_level": risk_level,
        "url_analysis": url_results,
    }