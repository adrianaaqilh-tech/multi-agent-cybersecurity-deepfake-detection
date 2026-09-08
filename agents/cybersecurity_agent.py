"""Cybersecurity Threat Detection Agent."""

import json

from config import llm
from tools.cybersecurity_tools import cybersecurity_indicator_checker


# ============================================================
# CYBERSECURITY PROMPT
# ============================================================

CYBERSECURITY_PROMPT = """
You are the Cybersecurity Threat Detection Agent.

Your responsibility is to analyse information extracted from ONE image
and the user's original request to identify possible cybersecurity threats.

The user request is the PRIMARY context for understanding what the user
wants to check. The OCR text and preliminary tool result are SUPPORTING
evidence only.

The image may contain:

- Banking messages
- Login requests
- Suspicious URLs
- Promotional offers
- Verification requests
- Account warnings
- Financial claims
- Social engineering messages
- Advertisements
- Other potentially suspicious content

Look for:

- Phishing
- Scams
- Suspicious URLs
- Malicious links
- Fraudulent advertisements
- Impersonation
- Social engineering
- Suspicious requests for personal information
- Suspicious financial or investment offers
- Urgency or pressure tactics
- Account suspension warnings
- Verification requests
- Login-related requests
- Claim or reward bait

Do not make unsupported conclusions.

A suspicious indicator does not automatically prove that something is malicious.

The risk score must represent the level of potential cybersecurity risk
based on the available evidence.

Risk score:
0 to 100

Risk level:
Low, Medium, or High

Confidence:
Low, Medium, or High


============================================================
FEW-SHOT EXAMPLES
============================================================

Example 1:

User request:
"Check this phishing email."

Image text:
"Your bank account will be suspended. Verify your account immediately
by clicking this link."

Expected reasoning:
The message contains account suspension pressure, urgency, verification
request, and a suspicious link. These indicators suggest potential phishing.

Example 2:

User request:
"Is this image related to a scam?"

Image text:
"Congratulations! You have won RM10,000. Click here immediately to claim
your reward."

Expected reasoning:
The reward claim, urgency, and request to click a link are suspicious
indicators commonly associated with scam messages.


============================================================
IMPORTANT ANALYSIS RULES
============================================================

- Analyse only the information available from the user request,
  image text, and preliminary tool result.
- Do not invent URLs, organisations, people, or threats.
- Do not assume that every banking message is malicious.
- Do not assume that every URL is malicious.
- Multiple suspicious indicators can increase the risk score.
- Explain why the identified indicators contribute to the assessment.
- Clearly state uncertainty when evidence is limited.
- Do not claim certainty without sufficient evidence.
- If the user specifically asks about phishing, scams, fraud, malicious
  links, or cybersecurity threats, consider that intent when assessing
  the potential risk.
- Use the preliminary cybersecurity tool result as supporting evidence.
- Do not automatically assign a risk score of 0 when cybersecurity
  indicators are present.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY one valid JSON object.

Keep the response concise.

Limit findings to 5 items.

Limit recommended_action to 3 items.

Keep analysis to 1-2 sentences.

Do not use Markdown.

Do not use code fences.

Do not include explanations outside the JSON object.

Use exactly these fields:

{
    "agent": "Cybersecurity Threat Detection Agent",
    "status": "Assessment status",
    "findings": [],
    "analysis": "Reasoning behind the assessment",
    "risk_score": 0,
    "risk_level": "Low",
    "confidence": "Medium",
    "recommended_action": []
}

The risk_score must be a number from 0 to 100.

The risk_level must be one of:

- Low
- Medium
- High

The confidence must be one of:

- Low
- Medium
- High
"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _get_tool_risk_score(
    tool_result: dict,
) -> float:
    """Convert deterministic tool indicators into a numeric risk score."""

    indicator_count = tool_result.get(
        "indicator_count",
        0,
    )

    if indicator_count >= 3:
        return 80.0

    if indicator_count >= 1:
        return 50.0

    return 10.0


def _normalise_llm_content(
    content,
) -> str:
    """Convert the LLM response into a clean JSON string."""

    if isinstance(content, list):

        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    content = str(
        content
    ).strip()

    if content.startswith(
        "```json"
    ):

        content = content[7:]

    elif content.startswith(
        "```"
    ):

        content = content[3:]

    if content.endswith(
        "```"
    ):

        content = content[:-3]

    content = content.strip()

    json_start = content.find(
        "{"
    )

    json_end = content.rfind(
        "}"
    )

    if (
        json_start != -1
        and json_end != -1
        and json_end > json_start
    ):

        content = content[
            json_start:
            json_end + 1
        ]

    return content.strip()


def _validate_result(
    parsed_result: dict,
) -> None:
    """Validate the required fields and values returned by the LLM."""

    required_fields = {
        "agent",
        "status",
        "findings",
        "analysis",
        "risk_score",
        "risk_level",
        "confidence",
        "recommended_action",
    }

    if not required_fields.issubset(
        parsed_result.keys()
    ):

        raise ValueError(
            "Missing required JSON fields."
        )

    if not isinstance(
        parsed_result["risk_score"],
        (int, float),
    ):

        raise ValueError(
            "Risk score must be numeric."
        )

    if not 0 <= parsed_result[
        "risk_score"
    ] <= 100:

        raise ValueError(
            "Risk score must be between 0 and 100."
        )

    if parsed_result[
        "risk_level"
    ] not in {
        "Low",
        "Medium",
        "High",
    }:

        raise ValueError(
            "Invalid risk level."
        )

    if parsed_result[
        "confidence"
    ] not in {
        "Low",
        "Medium",
        "High",
    }:

        raise ValueError(
            "Invalid confidence level."
        )


def _calculate_risk_level(
    risk_score: float,
) -> str:
    """Convert a numeric risk score into a risk level."""

    if risk_score >= 70:
        return "High"

    if risk_score >= 40:
        return "Medium"

    return "Low"


def _deterministic_cybersecurity_result(
    tool_result: dict,
) -> dict:
    """
    Create a cybersecurity result without using the LLM.

    This function is used when the LLM service is unavailable
    or when the LLM response cannot be processed.
    """

    indicator_count = tool_result.get(
        "indicator_count",
        0,
    )

    findings = tool_result.get(
        "indicators",
        [],
    )

    # Human developer modification:
    # The deterministic tool result provides a fast fallback
    # when the LLM service is unavailable.

    if indicator_count >= 3:

        risk_score = 80
        risk_level = "High"
        status = "Potential Threat Detected"

    elif indicator_count >= 1:

        risk_score = 50
        risk_level = "Medium"
        status = "Potential Threat Detected"

    else:

        risk_score = 10
        risk_level = "Low"
        status = "No Significant Threat Detected"

    confidence = "Low"

    if findings:

        analysis = (
            "The image contains indicators that may be associated "
            "with a cybersecurity threat. The assessment is based "
            "on the available image content and identified "
            "cybersecurity indicators."
        )

    else:

        analysis = (
            "No significant cybersecurity indicators were identified "
            "from the available image content."
        )

    recommended_action = [
        "Do not click suspicious links.",
        "Do not provide sensitive information.",
        "Verify the information using an official source.",
    ]

    return {
        "agent":
            "Cybersecurity Threat Detection Agent",

        "status":
            status,

        "findings":
            findings,

        "analysis":
            analysis,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "confidence":
            confidence,

        "recommended_action":
            recommended_action,
    }


# ============================================================
# CYBERSECURITY AGENT
# ============================================================

def cybersecurity_agent(
    image_text: str,
    user_request: str = "",
    use_llm: bool = True,
) -> dict:
    """
    Analyse text extracted from an image for cybersecurity threats.

    Args:
        image_text: Text extracted from the submitted image.
        user_request: Original request provided by the user.

    Returns:
        Structured cybersecurity assessment.
    """

    # Human developer modification:
    # The deterministic cybersecurity tool performs an initial scan
    # before the LLM performs semantic analysis.

    try:

        tool_result = cybersecurity_indicator_checker(
            image_text
        )

    except Exception:

        # Human developer modification:
        # If the custom cybersecurity tool fails, continue with a safe
        # empty result instead of crashing the entire workflow.

        tool_result = {
            "indicator_count": 0,
            "indicators": [],
            "risk_level": "Low",
        }

    tool_risk_score = _get_tool_risk_score(
        tool_result
    )

    # Human developer modification:
    # Skip the LLM completely during fallback mode so an unavailable
    # LM Studio server cannot cause a timeout.
    if not use_llm:
        return _deterministic_cybersecurity_result(
            tool_result
        )

    # ========================================================
    # LLM PROMPT
    # ========================================================

    prompt = f"""
{CYBERSECURITY_PROMPT}

============================================================
USER ORIGINAL REQUEST
============================================================

{user_request}


============================================================
TEXT EXTRACTED FROM IMAGE
============================================================

{image_text}


============================================================
PRELIMINARY TOOL RESULT
============================================================

{tool_result}


============================================================
TASK
============================================================

Analyse the user's request, extracted image information, and
preliminary cybersecurity indicators.

The user request is the primary context.

Identify only the most relevant cybersecurity findings.

Maximum findings: 5.

Explain the reasoning briefly in 1-2 sentences.

Return ONLY the required JSON object.
"""

    # ========================================================
    # LLM CALL
    # ========================================================

    try:

        # Human developer modification:
        # Gemma performs semantic analysis using the original user
        # request together with OCR and deterministic tool evidence.

        response = llm.invoke(
            prompt
        )

        content = _normalise_llm_content(
            response.content
        )

        parsed_result = json.loads(
            content
        )

        _validate_result(
            parsed_result
        )

        # ====================================================
        # COMBINE TOOL AND LLM SCORES
        # ====================================================

        model_score = float(
            parsed_result[
                "risk_score"
            ]
        )

        # Human developer modification:
        # The final score cannot ignore strong deterministic evidence.
        # The stronger of the model score and tool score is retained.

        final_score = max(
            model_score,
            tool_risk_score,
        )

        final_level = _calculate_risk_level(
            final_score
        )

        return {
            "agent":
                "Cybersecurity Threat Detection Agent",

            "status":
                parsed_result[
                    "status"
                ],

            "findings":
                parsed_result[
                    "findings"
                ],

            "analysis":
                parsed_result[
                    "analysis"
                ],

            "risk_score":
                final_score,

            "risk_level":
                final_level,

            "confidence":
                parsed_result[
                    "confidence"
                ],

            "recommended_action":
                parsed_result[
                    "recommended_action"
                ],
        }

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):

        # Human developer modification:
        # If Gemma produces invalid JSON, use the deterministic
        # cybersecurity result as a structured fallback.

        return _deterministic_cybersecurity_result(
            tool_result
        )

    except Exception:

        # Human developer modification:
        # A final safety handler prevents unexpected agent errors
        # from crashing the MAS workflow.

        return _deterministic_cybersecurity_result(
            tool_result
        )