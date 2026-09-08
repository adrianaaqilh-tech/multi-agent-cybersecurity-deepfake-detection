"""Deepfake Detection Agent."""

import json

from config import llm
from tools.deepfake_tools import media_indicator_analyzer


# ============================================================
# DEEPFAKE PROMPT
# ============================================================

DEEPFAKE_PROMPT = """
You are the Deepfake Detection Agent.

Your responsibility is to analyse ONE submitted image for possible
deepfake or AI-generated manipulation indicators.

The image may contain:

- People
- Faces
- Advertisements
- Banking messages
- Social media posts
- Promotional content
- Screenshots
- Claims involving a person's identity
- Other visual content


============================================================
DEEPFAKE INDICATORS
============================================================

Look for:

- Unnatural facial features
- Inconsistent facial boundaries
- Inconsistent lighting
- Unusual visual details
- Image artefacts
- Unnatural proportions
- Unusual skin or texture patterns
- Unusual image brightness
- Low visual sharpness
- Low edge detail
- Unusual image contrast
- Other possible visual manipulation indicators


============================================================
COMPLEX SCENARIO
============================================================

Consider the following scenario when analysing the image:

A banking-related advertisement contains a person's face and claims
that the person is an official representative promoting or endorsing
a financial service. The image may have been edited or generated to
make the person appear to support the message.

Possible deepfake indicators may include:

- Unnatural facial features
- Inconsistent facial boundaries
- Inconsistent lighting
- Unusual skin or texture patterns
- Unusual proportions
- Visual artefacts
- Unusual brightness or contrast
- Low sharpness or edge detail

The presence of these indicators may increase the potential
deepfake risk.

However, the agent must distinguish between:

1. Possible visual indicators
2. Possible manipulation
3. Confirmed deepfake content

Do not claim that the image is definitely a deepfake based only
on preliminary visual indicators.


============================================================
IMPORTANT LIMITATIONS
============================================================

- Do not claim that an image is definitely a deepfake.
- One visual indicator alone is not sufficient to confirm manipulation.
- Poor image quality alone does not prove manipulation.
- Low sharpness alone does not prove a deepfake.
- Unusual brightness alone does not prove a deepfake.
- Unusual contrast alone does not prove a deepfake.
- The absence of indicators does not prove authenticity.
- Do not fabricate visual evidence.
- Clearly distinguish observations from conclusions.
- State uncertainty when the available evidence is limited.


============================================================
FEW-SHOT EXAMPLE 1
============================================================

Input:

An image shows unnatural facial boundaries around a person's face
and inconsistent lighting between the face and the surrounding
environment.

Output:

{
    "agent": "Deepfake Detection Agent",
    "status": "Suspicious",
    "findings": [
        "Unnatural facial boundaries",
        "Inconsistent lighting"
    ],
    "analysis": "The visual inconsistencies may indicate image manipulation or AI-generated content, but these indicators alone are not sufficient to confirm a deepfake.",
    "risk_score": 70,
    "risk_level": "Medium",
    "confidence": "Medium",
    "recommended_action": [
        "Check the original source",
        "Perform additional authenticity verification"
    ]
}


============================================================
FEW-SHOT EXAMPLE 2
============================================================

Input:

An image shows natural facial features, consistent lighting,
normal proportions, and no obvious visual artefacts.

Output:

{
    "agent": "Deepfake Detection Agent",
    "status": "No Significant Indicators Detected",
    "findings": [],
    "analysis": "The analysed image does not show significant visual indicators commonly associated with deepfake manipulation. However, the absence of detected indicators does not prove that the image is authentic.",
    "risk_score": 20,
    "risk_level": "Low",
    "confidence": "Medium",
    "recommended_action": [
        "Verify the original source if authenticity is important"
    ]
}


============================================================
OUTPUT FORMAT
============================================================

Return ONLY one valid JSON object.

Keep the response concise. Limit findings to 5 items and recommended_action to 3 items. Keep analysis to 1-2 sentences.

Do not use Markdown.

Do not use code fences.

Do not include explanations outside the JSON object.

Use exactly these fields:

{
    "agent": "Deepfake Detection Agent",
    "status": "Assessment status",
    "findings": [],
    "analysis": "Reasoning behind the assessment",
    "risk_score": 0,
    "risk_level": "Low",
    "confidence": "Medium",
    "recommended_action": []
}

The risk_score must be a number from 0 to 100.

The risk_level must be:

- Low
- Medium
- High

The confidence must be:

- Low
- Medium
- High
"""


# ============================================================
# DEEPFAKE AGENT
# ============================================================

def deepfake_agent(
    image_path: str,
    use_llm: bool = True,
) -> dict:
    """
    Analyse an image for possible deepfake indicators.

    Args:
        image_path: File path of the submitted image.

    Returns:
        Structured Deepfake Agent assessment.
    """

    try:

        # Human developer modification:
        # The deterministic image-analysis tool performs the
        # initial technical inspection before LLM analysis.
        tool_result = media_indicator_analyzer(
            image_path
        )

        # Human developer modification:
        # Skip the LLM completely during fallback mode and return a
        # deterministic assessment from the image-analysis tool.
        if not use_llm:
            risk_score = float(tool_result.get("risk_score", 0))
            findings = tool_result.get("indicators", [])

            if risk_score >= 70:
                risk_level = "High"
                status = "Suspicious"
            elif risk_score >= 40:
                risk_level = "Medium"
                status = "Potential Manipulation"
            else:
                risk_level = "Low"
                status = "No Significant Indicators Detected"

            if findings:
                analysis = (
                    "The image contains visual indicators that may be "
                    "associated with image manipulation or deepfake content. "
                    "These indicators do not by themselves confirm that the "
                    "image is a deepfake."
                )
            else:
                analysis = (
                    "No significant visual indicators associated with "
                    "deepfake manipulation were identified. The absence of "
                    "indicators does not prove that the image is authentic."
                )

            return {
                "agent": "Deepfake Detection Agent",
                "status": status,
                "findings": findings,
                "analysis": analysis,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "confidence": "Low",
                "recommended_action": [
                    "Check the original source.",
                    "Perform additional authenticity verification.",
                ],
            }

        # ----------------------------------------------------
        # LLM PROMPT
        # ----------------------------------------------------

        prompt = f"""
{DEEPFAKE_PROMPT}

============================================================
PRELIMINARY IMAGE ANALYSIS
============================================================

{tool_result}


============================================================
TASK
============================================================

Analyse the image using the available technical indicators.

Identify only the most relevant possible manipulation indicators (maximum 5).

Explain the reasoning briefly in 1-2 sentences.

Distinguish possible indicators from confirmed manipulation.

Return ONLY the required JSON object.
"""

        # Human developer modification:
        # Gemma performs semantic interpretation of the
        # deterministic visual-analysis results.
        response = llm.invoke(
            prompt
        )

        content = response.content

        # ----------------------------------------------------
        # NORMALISE LLM RESPONSE
        # ----------------------------------------------------

        if isinstance(
            content,
            list,
        ):

            content = "".join(
                item.get(
                    "text",
                    "",
                )
                for item in content
                if isinstance(
                    item,
                    dict,
                )
            )

        content = str(
            content
        ).strip()

        # ----------------------------------------------------
        # REMOVE CODE FENCES
        # ----------------------------------------------------

        if content.startswith(
            "```json"
        ):

            content = content[
                7:
            ]

        elif content.startswith(
            "```"
        ):

            content = content[
                3:
            ]

        if content.endswith(
            "```"
        ):

            content = content[
                :-3
            ]

        content = content.strip()

        # ----------------------------------------------------
        # EXTRACT JSON OBJECT
        # ----------------------------------------------------

        # Human developer modification:
        # If Gemma adds text around the JSON object, extract
        # the actual JSON before attempting validation.
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

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        parsed_result = json.loads(
            content
        )

        # ----------------------------------------------------
        # REQUIRED FIELDS
        # ----------------------------------------------------

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

        # Human developer modification:
        # Validate the complete structured response before
        # allowing it into the MAS workflow.
        if not required_fields.issubset(
            parsed_result.keys()
        ):

            raise ValueError(
                "Missing required JSON fields."
            )

        # ----------------------------------------------------
        # VALIDATE RISK SCORE
        # ----------------------------------------------------

        if not isinstance(
            parsed_result[
                "risk_score"
            ],
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

        # ----------------------------------------------------
        # VALIDATE RISK LEVEL
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # VALIDATE CONFIDENCE
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # COMBINE TOOL + LLM RESULTS
        # ----------------------------------------------------

        model_risk_score = float(
            parsed_result[
                "risk_score"
            ]
        )

        tool_risk_score = float(
            tool_result.get(
                "risk_score",
                0,
            )
        )

        # Human developer modification:
        # Retain the stronger score between measurable technical
        # indicators and the LLM interpretation.
        risk_score = max(
            tool_risk_score,
            model_risk_score,
        )

        risk_score = max(
            0,
            min(
                risk_score,
                100,
            ),
        )

        # Prefer technical findings when available.
        findings = tool_result.get(
            "indicators",
            [],
        )

        if not findings:

            findings = parsed_result.get(
                "findings",
                [],
            )

        # ----------------------------------------------------
        # RECALCULATE RISK LEVEL
        # ----------------------------------------------------

        # Human developer modification:
        # Recalculate the displayed risk category from the
        # final numerical score to keep both values consistent.
        if risk_score >= 70:

            risk_level = "High"

            status = "Suspicious"

        elif risk_score >= 40:

            risk_level = "Medium"

            status = "Potential Manipulation"

        else:

            risk_level = "Low"

            status = (
                "No Significant Indicators Detected"
            )

        confidence = parsed_result[
            "confidence"
        ]

        analysis = parsed_result[
            "analysis"
        ]

        recommended_action = parsed_result[
            "recommended_action"
        ]

        return {
            "agent":
                "Deepfake Detection Agent",

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

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):

        # Human developer modification:
        # Invalid LLM output does not stop the workflow.
        # The deterministic image-analysis tool becomes the
        # silent fallback.
        try:

            tool_result = media_indicator_analyzer(
                image_path
            )

        except Exception:

            tool_result = {}

        risk_score = float(
            tool_result.get(
                "risk_score",
                0,
            )
        )

        # ----------------------------------------------------
        # FALLBACK RISK LEVEL
        # ----------------------------------------------------

        if risk_score >= 70:

            risk_level = "High"

            status = "Suspicious"

        elif risk_score >= 40:

            risk_level = "Medium"

            status = "Potential Manipulation"

        else:

            risk_level = "Low"

            status = (
                "No Significant Indicators Detected"
            )

        findings = tool_result.get(
            "indicators",
            [],
        )

        # ----------------------------------------------------
        # FALLBACK ANALYSIS
        # ----------------------------------------------------

        # IMPORTANT:
        # Do not expose JSON parsing errors or internal
        # fallback information to the user.
        if findings:

            analysis = (
                "The image contains visual indicators that "
                "may be associated with image manipulation "
                "or deepfake content. These indicators do "
                "not by themselves confirm that the image "
                "is a deepfake."
            )

        else:

            analysis = (
                "No significant visual indicators associated "
                "with deepfake manipulation were identified. "
                "However, the absence of detected indicators "
                "does not prove that the image is authentic."
            )

        recommended_action = [
            "Check the original source of the image.",
            "Perform additional authenticity verification.",
            "Do not treat the image as authentic based only on this analysis.",
        ]

        return {
            "agent":
                "Deepfake Detection Agent",

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
                "Low",

            "recommended_action":
                recommended_action,
        }

    except Exception:

        # Human developer modification:
        # Final safety handler prevents an unexpected Deepfake
        # Agent failure from crashing the entire MAS.
        return {
            "agent":
                "Deepfake Detection Agent",

            "status":
                "Analysis Unavailable",

            "findings":
                [],

            "analysis":
                (
                    "The deepfake analysis could "
                    "not be completed."
                ),

            "risk_score":
                0,

            "risk_level":
                "Low",

            "confidence":
                "Low",

            "recommended_action":
                [
                    "Review the image manually."
                ],
        }