"""Evidence Verification Agent."""

import json

from config import llm
from tools.evidence_tools import evidence_consistency_checker


# ============================================================
# EVIDENCE VERIFICATION PROMPT
# ============================================================

EVIDENCE_PROMPT = """
You are the Evidence Verification Agent.

Your responsibility is to evaluate the consistency and reliability
of findings produced by the Cybersecurity Threat Detection Agent
and the Deepfake Detection Agent for ONE image.

You do NOT perform the primary cybersecurity or deepfake detection.

Instead, you act as an independent verification layer that evaluates
the findings produced by the specialist agents.

One or both specialist agents may have been selected by the Dynamic
Router.

If both specialists were selected, evaluate BOTH sets of findings.

The Evidence Agent must focus only on the available findings and
evidence. It must not invent findings from an agent that was not
selected.


============================================================
YOUR RESPONSIBILITIES
============================================================

You must:

- Evaluate the available specialist findings.
- Identify agreements between available evidence and findings.
- Identify disagreements when they exist.
- Consider the available technical indicators.
- Identify missing or insufficient evidence.
- Determine how strongly the findings are supported.
- Assess the reliability of the available evidence.
- Clearly state uncertainty.
- Do not automatically assume a specialist agent is correct.
- Do not fabricate evidence.
- Do not fabricate external sources.
- Do not claim that an assessment is confirmed when evidence is
  insufficient.
- When both specialists are available, evaluate their findings
  independently and together.


============================================================
CYBERSECURITY EVIDENCE
============================================================

When Cybersecurity findings are available, consider evidence such as:

- Verification requests
- Urgency language
- Account suspension warnings
- Login-related requests
- Banking-related content
- Suspicious login URL patterns
- Suspicious links
- Unexpected attachments
- Suspicious sender information
- Social engineering indicators
- Other cybersecurity indicators

Determine whether the available evidence supports the cybersecurity
assessment.

A high cybersecurity risk does not automatically prove that every
claim in the image is malicious.


============================================================
DEEPFAKE EVIDENCE
============================================================

When Deepfake findings are available, consider evidence such as:

- Unusual facial features
- Inconsistent facial boundaries
- Unusual lighting
- Unusual skin or texture patterns
- Unusual proportions
- Visual artefacts
- Unusual brightness or contrast
- Low visual sharpness
- Low edge detail

Determine whether the available evidence supports the deepfake
assessment.

A high deepfake risk does not automatically prove that the image
is a confirmed deepfake.


============================================================
WHEN BOTH SPECIALISTS ARE AVAILABLE
============================================================

Evaluate Cybersecurity and Deepfake findings separately.

For example:

Cybersecurity:
- Check whether the detected cybersecurity indicators support
  the cybersecurity risk assessment.

Deepfake:
- Check whether the detected visual indicators support
  the deepfake risk assessment.

Then determine whether the combined evidence provides:

- Strong support
- Partial support
- Weak support
- Insufficient support

Do not assume that one specialist confirms the other.

Cybersecurity evidence does not prove that an image is a deepfake.

Deepfake evidence does not prove that a message is a phishing scam.


============================================================
EVIDENCE RELIABILITY SCORE
============================================================

The evidence reliability score must be between 0 and 100.

0 means:

The available evidence is insufficient or unreliable.

100 means:

The available evidence strongly supports the specialist findings.

The score represents the strength of the AVAILABLE EVIDENCE.

It does NOT represent absolute truth.


============================================================
SUPPORT LEVEL
============================================================

Use one of:

- Strong
- Partial
- Weak
- Insufficient


============================================================
CONFIDENCE
============================================================

Use one of:

- Low
- Medium
- High


============================================================
IMPORTANT RULES
============================================================

- Do not automatically trust specialist agents.
- Compare findings objectively.
- Distinguish agreement from confirmation.
- Identify missing evidence.
- State uncertainty when appropriate.
- Do not fabricate verification results.
- Do not fabricate sources.
- Do not claim that an image is authentic or fake without
  sufficient evidence.
- Do not create findings for an unselected specialist agent.
- Do not treat the presence of two agents as automatic proof.
- Evidence verification must be based on the evidence actually
  produced by the specialist agents.


============================================================
OUTPUT FORMAT
============================================================

Return ONLY one valid JSON object.

Keep the response concise.

Limit findings to 3 items.

Limit recommended_action to 3 items.

Keep analysis to 1 sentence.

Do not use Markdown.

Do not use code fences.

Do not include explanations outside the JSON object.

Use exactly these fields:

{
    "agent": "Evidence Verification Agent",
    "status": "Verification status",
    "findings": [],
    "analysis": "Reasoning behind the verification",
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
# HELPER FUNCTIONS
# ============================================================

def _normalise_llm_content(content) -> str:
    """
    Convert the LLM response into a clean JSON string.
    """

    if isinstance(content, list):

        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    content = str(content).strip()

    if content.startswith("```json"):

        content = content[7:]

    elif content.startswith("```"):

        content = content[3:]

    if content.endswith("```"):

        content = content[:-3]

    content = content.strip()

    json_start = content.find("{")
    json_end = content.rfind("}")

    if (
        json_start != -1
        and json_end != -1
        and json_end > json_start
    ):

        content = content[
            json_start:json_end + 1
        ]

    return content.strip()


def _validate_result(
    parsed_result: dict,
) -> None:
    """
    Validate the structure and values returned by the Evidence Agent.
    """

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
    reliability_score: float,
) -> str:
    """
    Convert evidence reliability into a risk level.
    """

    if reliability_score >= 70:
        return "High"

    if reliability_score >= 40:
        return "Medium"

    return "Low"


def _calculate_support_level(
    reliability_score: float,
) -> str:
    """
    Convert evidence reliability into a support level.
    """

    if reliability_score >= 75:
        return "Strong"

    if reliability_score >= 50:
        return "Partial"

    if reliability_score >= 30:
        return "Weak"

    return "Insufficient"


# ============================================================
# DETERMINISTIC EVIDENCE FALLBACK
# ============================================================

def _deterministic_evidence_result(
    cybersecurity_result: dict,
    deepfake_result: dict,
    tool_result: dict,
    cyber_selected: bool,
    deepfake_selected: bool,
    selected_agent: str,
    cybersecurity_findings: list,
    deepfake_findings: list,
    cybersecurity_score: float,
    deepfake_score: float,
) -> dict:
    """
    Create an evidence verification result without using the LLM.

    This provides a fast fallback when the local LLM service
    is unavailable.
    """

    reliability_score = float(
        tool_result.get(
            "reliability_score",
            20,
        )
    )

    reliability_score = min(
        reliability_score,
        100,
    )

    support_level = _calculate_support_level(
        reliability_score
    )

    risk_level = _calculate_risk_level(
        reliability_score
    )

    if reliability_score >= 75:

        verification_status = "Supported"

    elif reliability_score >= 50:

        verification_status = "Partially Supported"

    elif reliability_score >= 30:

        verification_status = "Not Supported"

    else:

        verification_status = "Cannot Verify"

    findings = tool_result.get(
        "issues",
        [],
    )

    if not findings:

        if cyber_selected and deepfake_selected:

            findings = [
                "Evidence from both specialist agents was reviewed."
            ]

        elif cyber_selected:

            findings = [
                "Cybersecurity specialist evidence was reviewed."
            ]

        elif deepfake_selected:

            findings = [
                "Deepfake specialist evidence was reviewed."
            ]

        else:

            findings = [
                "No specialist evidence is available for verification."
            ]

    findings = findings[:3]

    if reliability_score >= 50:

        analysis = (
            "The available evidence provides some support for "
            "the specialist findings, but the assessment should "
            "not be treated as definitive."
        )

    else:

        analysis = (
            "The available evidence is limited, so the specialist "
            "findings cannot be verified with high confidence."
        )

    confidence = "Low"

    recommended_action = [
        "Review the original image source.",
        "Verify important claims using reliable sources.",
        "Do not rely on a single automated assessment.",
    ]

    return {
        "agent":
            "Evidence Verification Agent",

        "status":
            verification_status,

        "findings":
            findings,

        "analysis":
            analysis,

        "risk_score":
            reliability_score,

        "risk_level":
            risk_level,

        "confidence":
            confidence,

        "recommended_action":
            recommended_action,

        "verification_status":
            verification_status,

        "support_level":
            support_level,

        "evidence_reliability_score":
            reliability_score,

        "agent_findings_consistency":
            tool_result.get(
                "agent_consistency",
                "Unknown",
            ),

        "cybersecurity_score":
            cybersecurity_score,

        "deepfake_score":
            deepfake_score,

        "cybersecurity_findings":
            cybersecurity_findings,

        "deepfake_findings":
            deepfake_findings,
    }


# ============================================================
# EVIDENCE AGENT
# ============================================================

def evidence_agent(
    cybersecurity_result: dict,
    deepfake_result: dict,
    use_llm: bool = True,
) -> dict:
    """
    Verify the consistency and reliability of specialist findings.

    Args:
        cybersecurity_result:
            Structured result produced by the Cybersecurity Agent.
            It may be empty when cybersecurity was not selected.

        deepfake_result:
            Structured result produced by the Deepfake Agent.
            It may be empty when deepfake was not selected.

    Returns:
        A structured evidence verification assessment.
    """

    cybersecurity_result = (
        cybersecurity_result or {}
    )

    deepfake_result = (
        deepfake_result or {}
    )

    # ========================================================
    # DETERMINE ACTIVE SPECIALISTS
    # ========================================================

    cyber_selected = bool(
        cybersecurity_result
    )

    deepfake_selected = bool(
        deepfake_result
    )

    # ========================================================
    # RUN DETERMINISTIC EVIDENCE TOOL
    # ========================================================

    try:

        # Human developer modification:
        # The deterministic evidence tool performs an initial
        # consistency check before LLM-based verification.

        tool_result = evidence_consistency_checker(
            cybersecurity_result,
            deepfake_result,
        )

    except Exception:

        # Human developer modification:
        # If the evidence tool fails, provide a safe fallback
        # instead of stopping the complete MAS workflow.

        tool_result = {
            "selected_agent":
                "Unknown",

            "selected_score":
                0,

            "selected_findings":
                [],

            "cybersecurity_score":
                float(
                    cybersecurity_result.get(
                        "risk_score",
                        0,
                    )
                ),

            "deepfake_score":
                float(
                    deepfake_result.get(
                        "risk_score",
                        0,
                    )
                ),

            "cybersecurity_findings":
                cybersecurity_result.get(
                    "findings",
                    [],
                ),

            "deepfake_findings":
                deepfake_result.get(
                    "findings",
                    [],
                ),

            "agent_consistency":
                "Insufficient",

            "support_level":
                "Insufficient",

            "reliability_score":
                20,

            "issues":
                [
                    "Evidence consistency analysis was unavailable."
                ],
        }

    # ========================================================
    # DETERMINE ACTIVE AGENT DESCRIPTION
    # ========================================================

    if (
        cyber_selected
        and deepfake_selected
    ):

        selected_agent = (
            "Cybersecurity Threat Detection Agent "
            "+ Deepfake Detection Agent"
        )

    elif cyber_selected:

        selected_agent = (
            "Cybersecurity Threat Detection Agent"
        )

    elif deepfake_selected:

        selected_agent = (
            "Deepfake Detection Agent"
        )

    else:

        selected_agent = "Unknown"

    # ========================================================
    # PREPARE SPECIALIST RESULTS
    # ========================================================

    cybersecurity_findings = (
        cybersecurity_result.get(
            "findings",
            [],
        )
    )

    deepfake_findings = (
        deepfake_result.get(
            "findings",
            [],
        )
    )

    cybersecurity_score = float(
        cybersecurity_result.get(
            "risk_score",
            0,
        )
    )

    deepfake_score = float(
        deepfake_result.get(
            "risk_score",
            0,
        )
    )

    # Human developer modification:
    # Skip the LLM during fallback mode and use the deterministic
    # evidence checker so the failure path remains fast.
    if not use_llm:
        return _deterministic_evidence_result(
            cybersecurity_result,
            deepfake_result,
            tool_result,
            cyber_selected,
            deepfake_selected,
            selected_agent,
            cybersecurity_findings,
            deepfake_findings,
            cybersecurity_score,
            deepfake_score,
        )

    # ========================================================
    # BUILD LLM EVIDENCE PROMPT
    # ========================================================

    prompt = f"""
{EVIDENCE_PROMPT}

============================================================
ACTIVE SPECIALIST AGENTS
============================================================

{selected_agent}


============================================================
CYBERSECURITY SPECIALIST RESULT
============================================================

{cybersecurity_result}


============================================================
DEEPFAKE SPECIALIST RESULT
============================================================

{deepfake_result}


============================================================
DETERMINISTIC EVIDENCE TOOL RESULT
============================================================

{tool_result}


============================================================
TASK
============================================================

Verify the available specialist findings.

If only one specialist result is available, verify that specialist.

If both specialist results are available, evaluate BOTH specialists
and the combined evidence.

Identify only the most important verification points.

Maximum findings: 3.

Consider:

- agreements
- disagreements
- missing evidence
- consistency
- limitations
- support for the specialist findings

Do not create findings that are not present in the specialist results.

Return ONLY the required JSON object.
"""

    # ========================================================
    # LLM VERIFICATION
    # ========================================================

    try:

        # Human developer modification:
        # The local LLM independently evaluates the available
        # specialist evidence instead of directly accepting it.

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
        # COMBINE LLM AND DETERMINISTIC EVIDENCE
        # ====================================================

        model_score = float(
            parsed_result[
                "risk_score"
            ]
        )

        tool_score = float(
            tool_result.get(
                "reliability_score",
                0,
            )
        )

        # Human developer modification:
        # Use the deterministic evidence result as a lower-bound
        # safeguard so useful specialist evidence is not discarded
        # by an unexpectedly low LLM score.

        reliability_score = max(
            model_score,
            tool_score,
        )

        reliability_score = min(
            reliability_score,
            100,
        )

        support_level = _calculate_support_level(
            reliability_score
        )

        risk_level = _calculate_risk_level(
            reliability_score
        )

        findings = parsed_result.get(
            "findings",
            [],
        )[:3]

        analysis = parsed_result.get(
            "analysis",
            "",
        )

        confidence = parsed_result.get(
            "confidence",
            "Medium",
        )

        recommended_action = parsed_result.get(
            "recommended_action",
            [],
        )[:3]

        verification_status = (
            "Supported"
            if reliability_score >= 75
            else "Partially Supported"
            if reliability_score >= 50
            else "Not Supported"
            if reliability_score >= 30
            else "Cannot Verify"
        )

        return {
            "agent":
                "Evidence Verification Agent",

            "status":
                verification_status,

            "findings":
                findings,

            "analysis":
                analysis,

            "risk_score":
                reliability_score,

            "risk_level":
                risk_level,

            "confidence":
                confidence,

            "recommended_action":
                recommended_action,

            "verification_status":
                verification_status,

            "support_level":
                support_level,

            "evidence_reliability_score":
                reliability_score,

            "agent_findings_consistency":
                tool_result.get(
                    "agent_consistency",
                    "Unknown",
                ),

            "cybersecurity_score":
                cybersecurity_score,

            "deepfake_score":
                deepfake_score,

            "cybersecurity_findings":
                cybersecurity_findings,

            "deepfake_findings":
                deepfake_findings,
        }

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):

        # Human developer modification:
        # Invalid LLM output does not crash the workflow.
        # The deterministic evidence tool becomes the fallback.

        return _deterministic_evidence_result(
            cybersecurity_result,
            deepfake_result,
            tool_result,
            cyber_selected,
            deepfake_selected,
            selected_agent,
            cybersecurity_findings,
            deepfake_findings,
            cybersecurity_score,
            deepfake_score,
        )

    except Exception:

        # Human developer modification:
        # A final exception handler prevents an unexpected Evidence
        # Agent failure from crashing the complete MAS workflow.

        return _deterministic_evidence_result(
            cybersecurity_result,
            deepfake_result,
            tool_result,
            cyber_selected,
            deepfake_selected,
            selected_agent,
            cybersecurity_findings,
            deepfake_findings,
            cybersecurity_score,
            deepfake_score,
        )