"""Workflow graph for the Cybersecurity and Deepfake Multi-Agent System."""

from typing import Any
import textwrap
import time

from langgraph.graph import StateGraph, START, END

from config import is_llm_available

from agents.cybersecurity_agent import cybersecurity_agent
from agents.deepfake_agent import deepfake_agent
from agents.evidence_agent import evidence_agent
from agents.router import router_agent
from state import MASState
from tools.ocr_tools import extract_text_from_image


MAX_WORKFLOW_ITERATIONS = 10


def workflow_execution_guard(state: MASState) -> dict[str, Any]:
    """Prevent excessive workflow execution."""
    current = int(state.get("workflow_iterations", 0)) + 1

    if current > MAX_WORKFLOW_ITERATIONS:
        errors = list(state.get("errors", []))
        errors.append("Workflow execution limit exceeded.")

        return {
            "workflow_iterations": current,
            "errors": errors,
            "fatal_error": True,
            "fatal_error_message": (
                "Workflow execution stopped because the maximum "
                "allowed iterations was exceeded."
            ),
        }

    return {
        "workflow_iterations": current
    }


def route_after_execution_guard(state: MASState) -> str:
    """Determine whether the workflow can continue."""
    return "end" if state.get("fatal_error", False) else "continue"


def input_validation_node(state: MASState) -> dict[str, Any]:
    """Validate input and select the fast fallback when LLM is unavailable."""
    user_input = state.get("user_input", "")
    image_path = state.get("image_path", "")

    if not user_input:
        return {
            "errors": ["User request was not provided."],
            "fatal_error": True,
            "fatal_error_message": "User request was not provided.",
        }

    if not image_path:
        return {
            "errors": ["Image path was not provided."],
            "fatal_error": True,
            "fatal_error_message": "Image path was not provided.",
        }

    try:
        llm_available = is_llm_available()
    except Exception:
        llm_available = False

    if not llm_available:
        selected = _fallback_route_from_request(user_input)

        route = (
            "both"
            if set(selected) == {"cybersecurity", "deepfake"}
            else selected[0]
        )

        return {
            "errors": state.get("errors", []),
            "selected_agents": selected,
            "routing_reason": (
                "The AI service was unavailable, so deterministic "
                "fallback routing was used."
            ),
            "router_status": "FALLBACK",
            "route": route,
            "llm_unavailable": True,
            "image_text": "",
        }

    return {
        "errors": state.get("errors", []),
        "llm_unavailable": False,
    }


def route_after_input_validation(state: MASState) -> str:
    """Route input to error response, fallback, or normal OCR processing."""
    if state.get("fatal_error", False):
        return "error"

    if state.get("llm_unavailable", False):
        return "fallback"

    return "ocr"


def ocr_node(state: MASState) -> dict[str, Any]:
    """Extract text from the submitted image using OCR."""
    try:
        # Human developer modification: OCR runs before the Dynamic Router,
        # giving the Router supporting text from the submitted image.
        image_text = extract_text_from_image(
            state.get("image_path", "")
        )

        return {
            "image_text": image_text or ""
        }

    except Exception as error:
        errors = list(state.get("errors", []))
        errors.append(f"OCR failed: {error}")

        return {
            "errors": errors,
            "fatal_error": True,
            "fatal_error_message": (
                "The submitted image could not be processed. "
                "Please check the image path and file."
            ),
        }


def route_after_ocr(state: MASState) -> str:
    """Route OCR failures to an error response or continue to the Router."""
    return "error" if state.get("fatal_error", False) else "router"


def error_response_node(state: MASState) -> dict[str, Any]:
    """Prepare a fatal error state for the final user-facing error response."""
    return {
        "fatal_error": True,
        "router_status": "ERROR",
    }


def fallback_router_node(state: MASState) -> dict[str, Any]:
    """Route using deterministic user-intent matching without any LLM call."""
    selected = list(state.get("selected_agents", []))

    if not selected:
        selected = _fallback_route_from_request(
            state.get("user_input", "")
        )

    route = (
        "both"
        if set(selected) == {"cybersecurity", "deepfake"}
        else selected[0]
    )

    return {
        "selected_agents": selected,
        "routing_reason": state.get(
            "routing_reason",
            "Deterministic fallback routing was used."
        ),
        "router_status": "FALLBACK",
        "route": route,

        # Human developer modification: explicitly keep the entire
        # workflow in fallback mode so specialist agents do not
        # attempt additional LLM calls after the Router fails.
        "llm_unavailable": True,

        "image_text": "",
    }


# ============================================================
# DYNAMIC ROUTER
# ============================================================

def _fallback_route_from_request(user_input: str) -> list[str]:
    """Provide a local routing fallback when the LLM Router fails."""
    request = (user_input or "").lower()

    cyber = [
        "cybersecurity",
        "cyber security",
        "phishing",
        "phish",
        "phshing",
        "phising",
        "phisingg",
        "scam",
        "fraud",
        "malware",
        "ransomware",
        "hacking",
        "hack",
        "social engineering",
        "account security",
        "account verification",
        "cyber attack",
        "cyber threat",
        "security threat",
        "suspicious link",
        "suspicious url",
        "malicious link",
        "online security",
        "password",
        "login",
        "verify account",
    ]

    deepfake = [
        "deepfake",
        "deep fake",
        "ai generated",
        "ai-generated",
        "ai generated image",
        "ai-generated image",
        "fake image",
        "fake video",
        "manipulated image",
        "manipulated video",
        "image manipulation",
        "video manipulation",
        "face manipulation",
        "identity manipulation",
        "synthetic media",
        "visual authenticity",
        "edited image",
        "image authenticity",
    ]

    cyber_score = sum(
        keyword in request for keyword in cyber
    )

    deepfake_score = sum(
        keyword in request for keyword in deepfake
    )

    # Human developer modification: the fallback supports the required
    # Both path instead of forcing a single specialist.
    if cyber_score > 0 and deepfake_score > 0:
        return ["cybersecurity", "deepfake"]

    if cyber_score > 0:
        return ["cybersecurity"]

    if deepfake_score > 0:
        return ["deepfake"]

    return ["out_of_scope"]


def router_node(state: MASState) -> dict[str, Any]:
    """Run the LLM Dynamic Router using user intent and OCR context."""
    user_input = state.get("user_input", "")
    image_text = state.get("image_text", "")

    try:
        # Human developer modification: the original user prompt is the
        # primary routing signal; OCR is supplied only as supporting context.
        result = router_agent(
            user_input,
            image_text
        )

        selected = result.get(
            "selected_agents",
            []
        )

        reason = result.get(
            "routing_reason",
            "User request was used for routing."
        )

        status = result.get(
            "router_status",
            "SUCCESS"
        )

        if not isinstance(selected, list) or not selected:
            raise ValueError(
                "Dynamic Router returned an invalid agent list."
            )

        allowed = {
            "cybersecurity",
            "deepfake",
            "out_of_scope",
        }

        if any(agent not in allowed for agent in selected):
            raise ValueError(
                "Dynamic Router selected an invalid routing option."
            )

        if "out_of_scope" in selected and len(selected) != 1:
            raise ValueError(
                "out_of_scope cannot be combined with specialist agents."
            )

        if set(selected) == {
            "cybersecurity",
            "deepfake",
        }:
            selected = [
                "cybersecurity",
                "deepfake",
            ]

        elif selected not in [
            ["cybersecurity"],
            ["deepfake"],
            ["out_of_scope"],
        ]:
            raise ValueError(
                "Dynamic Router returned an unsupported route."
            )

        route = (
            "both"
            if set(selected) == {
                "cybersecurity",
                "deepfake",
            }
            else selected[0]
        )

        return {
            "selected_agents": selected,
            "routing_reason": reason,
            "router_status": status,
            "route": route,

            # Human developer modification: if the Router itself
            # reports FALLBACK, preserve fallback mode for all
            # downstream agents.
            "llm_unavailable": (
                True if status == "FALLBACK" else False
            ),
        }

    except Exception as error:
        errors = list(state.get("errors", []))
        errors.append(
            f"Dynamic Router Error: {error}"
        )

        fallback = _fallback_route_from_request(
            user_input
        )

        return {
            "selected_agents": fallback,
            "routing_reason": (
                "The specialist route was selected using "
                "the user request as a fallback."
            ),
            "router_status": "FALLBACK",
            "route": (
                "both"
                if set(fallback) == {
                    "cybersecurity",
                    "deepfake",
                }
                else fallback[0]
            ),

            # Human developer modification: Router failure must
            # propagate fallback status to all downstream agents.
            "llm_unavailable": True,

            "errors": errors,
        }


def route_after_router(state: MASState) -> list[str]:
    """Fan out to one specialist, both specialists, or final response."""
    if state.get("fatal_error", False):
        return ["final_response"]

    selected = list(
        state.get("selected_agents", [])
    )

    if selected == ["cybersecurity"]:
        return ["cybersecurity"]

    if selected == ["deepfake"]:
        return ["deepfake"]

    if set(selected) == {
        "cybersecurity",
        "deepfake",
    }:
        # Human developer modification: Both activates both specialist nodes.
        return [
            "cybersecurity",
            "deepfake",
        ]

    return ["final_response"]


# ============================================================
# SPECIALIST AGENTS
# ============================================================

def cybersecurity_node(state: MASState) -> dict[str, Any]:
    """Run the Cybersecurity Threat Detection Agent."""
    try:
        result = cybersecurity_agent(
            state.get("image_text", ""),
            state.get("user_input", ""),
            use_llm=not state.get(
                "llm_unavailable",
                False
            ),
        )

        return {
            "cybersecurity_result": result
        }

    except Exception as error:
        errors = list(state.get("errors", []))
        errors.append(
            f"Cybersecurity Agent Error: {error}"
        )

        return {
            "cybersecurity_result": {
                "agent": "Cybersecurity Threat Detection Agent",
                "status": "Analysis Unavailable",
                "findings": [],
                "analysis": (
                    "The cybersecurity assessment "
                    "could not be completed."
                ),
                "risk_score": 0,
                "risk_level": "Unknown",
                "confidence": "Low",
                "recommended_action": [
                    "Review the image manually."
                ],
            },
            "errors": errors,
        }


def deepfake_node(state: MASState) -> dict[str, Any]:
    """Run the Deepfake Detection Agent."""
    try:
        result = deepfake_agent(
            state.get("image_path", ""),
            use_llm=not state.get(
                "llm_unavailable",
                False
            ),
        )

        return {
            "deepfake_result": result
        }

    except Exception as error:
        errors = list(state.get("errors", []))
        errors.append(
            f"Deepfake Agent Error: {error}"
        )

        return {
            "deepfake_result": {
                "agent": "Deepfake Detection Agent",
                "status": "Analysis Unavailable",
                "findings": [],
                "analysis": (
                    "The deepfake assessment "
                    "could not be completed."
                ),
                "risk_score": 0,
                "risk_level": "Unknown",
                "confidence": "Low",
                "recommended_action": [
                    "Review the image manually."
                ],
            },
            "errors": errors,
        }


# ============================================================
# EVIDENCE AGENT
# ============================================================

def evidence_node(state: MASState) -> dict[str, Any]:
    """Run Evidence Verification on all active specialist results."""
    selected = list(
        state.get("selected_agents", [])
    )

    cyber_result = state.get(
        "cybersecurity_result",
        {}
    )

    deepfake_result = state.get(
        "deepfake_result",
        {}
    )

    if "cybersecurity" not in selected:
        cyber_result = {}

    if "deepfake" not in selected:
        deepfake_result = {}

    if not selected or selected == ["out_of_scope"]:
        return {
            "evidence_result": _empty_evidence()
        }

    try:
        # Human developer modification: Evidence receives both actual
        # specialist results when the Router selected Both.
        result = evidence_agent(
            cyber_result,
            deepfake_result,
            use_llm=not state.get(
                "llm_unavailable",
                False
            ),
        )

        return {
            "evidence_result": result
        }

    except Exception as error:
        errors = list(state.get("errors", []))
        errors.append(
            f"Evidence Agent Error: {error}"
        )

        return {
            "evidence_result": _empty_evidence(),
            "errors": errors,
        }


def _empty_evidence() -> dict[str, Any]:
    """Return a safe Evidence Agent fallback result."""
    return {
        "agent": "Evidence Verification Agent",
        "status": "Cannot Verify",
        "findings": [],
        "analysis": (
            "The available evidence could not be fully verified."
        ),
        "risk_score": 0,
        "risk_level": "Unknown",
        "confidence": "Low",
        "recommended_action": [
            "Review the evidence manually."
        ],
        "verification_status": "Cannot Verify",
        "support_level": "Insufficient",
        "evidence_reliability_score": 0,
        "agent_findings_consistency": "Unknown",
    }


# ============================================================
# FINAL ASSESSMENT
# ============================================================

def _safe_score(
    result: dict[str, Any],
    key: str = "risk_score",
) -> float:
    """Safely convert a result score into a float."""
    try:
        return float(
            result.get(key, 0)
        )
    except (TypeError, ValueError):
        return 0.0


def _calculate_overall_score(
    cyber: dict[str, Any],
    deepfake: dict[str, Any],
    selected: list[str],
) -> float:
    """Calculate the overall score from all active specialists."""
    scores = []

    if "cybersecurity" in selected:
        scores.append(
            _safe_score(cyber)
        )

    if "deepfake" in selected:
        scores.append(
            _safe_score(deepfake)
        )

    return round(
        max(scores),
        2
    ) if scores else 0.0


def _get_risk_level(score: float) -> str:
    """Convert a score to Low, Medium, or High."""
    if score >= 70:
        return "High"

    if score >= 40:
        return "Medium"

    return "Low"


def _get_confidence(
    cyber: dict[str, Any],
    deepfake: dict[str, Any],
    selected: list[str],
) -> str:
    """Combine confidence levels from active specialists."""
    values = []

    if "cybersecurity" in selected:
        values.append(
            str(
                cyber.get(
                    "confidence",
                    "Low"
                )
            ).lower()
        )

    if "deepfake" in selected:
        values.append(
            str(
                deepfake.get(
                    "confidence",
                    "Low"
                )
            ).lower()
        )

    if "high" in values:
        return "High"

    if "medium" in values:
        return "Medium"

    return "Low"


def _combine_confidence(
    specialist: str,
    evidence_score: float,
) -> str:
    """Combine specialist confidence with evidence reliability."""
    confidence = (
        specialist or "Low"
    ).strip().lower()

    if confidence == "high" and evidence_score >= 75:
        return "High"

    if confidence in {"high", "medium"} and evidence_score >= 50:
        return "Medium"

    if confidence == "low" and evidence_score >= 75:
        return "Medium"

    return "Low"


def _collect_findings(
    cyber: dict[str, Any],
    deepfake: dict[str, Any],
    selected: list[str],
) -> list[str]:
    """Collect unique findings from every active specialist."""
    findings = []

    for result in (
        cyber if "cybersecurity" in selected else {},
        deepfake if "deepfake" in selected else {},
    ):
        values = result.get(
            "findings",
            []
        )

        if not isinstance(values, list):
            continue

        for value in values:
            text = str(value).strip()

            if text and text not in findings:
                findings.append(text)

    return findings


def _build_overall_analysis(
    cyber: dict[str, Any],
    deepfake: dict[str, Any],
    evidence: dict[str, Any],
    score: float,
    risk: str,
    selected: list[str],
) -> str:
    """Build the final analysis from active specialist results and evidence."""

    if set(selected) == {
        "cybersecurity",
        "deepfake",
    }:
        summary = (
            f"The combined assessment is {risk} "
            f"with an overall score of {score}% ."
        )

    elif selected == ["cybersecurity"]:
        summary = (
            f"The cybersecurity assessment is {risk} "
            f"with a risk score of {score}%."
        )

    elif selected == ["deepfake"]:
        summary = (
            f"The deepfake assessment is {risk} "
            f"with a risk score of {score}%."
        )

    else:
        return (
            "The request is outside the supported cybersecurity "
            "and deepfake analysis scope."
        )

    analyses = []

    if (
        "cybersecurity" in selected
        and cyber.get("analysis")
    ):
        analyses.append(
            f"Cybersecurity: {cyber['analysis']}"
        )

    if (
        "deepfake" in selected
        and deepfake.get("analysis")
    ):
        analyses.append(
            f"Deepfake: {deepfake['analysis']}"
        )

    text = summary + (
        " " + " ".join(analyses)
        if analyses
        else ""
    )

    evidence_score = _safe_score(
        evidence,
        "evidence_reliability_score"
    )

    if evidence_score >= 75:
        text += (
            " The available evidence strongly "
            "supports the assessment."
        )

    elif evidence_score >= 50:
        text += (
            " The available evidence provides "
            "partial support for the assessment."
        )

    elif selected:
        text += (
            " The available evidence is limited, "
            "so the assessment should be interpreted with caution."
        )

    return text


def _get_routing_path(
    selected: list[str],
) -> str:
    """Return a readable routing path."""

    if set(selected) == {
        "cybersecurity",
        "deepfake",
    }:
        return "CYBERSECURITY + DEEPFAKE ANALYSIS"

    if selected == ["cybersecurity"]:
        return "CYBERSECURITY ANALYSIS"

    if selected == ["deepfake"]:
        return "DEEPFAKE ANALYSIS"

    return "OUT OF SCOPE"


def final_response_node(
    state: MASState,
) -> dict[str, Any]:
    """Generate the final user-facing assessment."""

    start = state.get(
        "_start_time",
        time.perf_counter()
    )

    selected = list(
        state.get(
            "selected_agents",
            []
        )
    )

    cyber = state.get(
        "cybersecurity_result",
        {}
    )

    deepfake = state.get(
        "deepfake_result",
        {}
    )

    evidence = state.get(
        "evidence_result",
        {}
    )

    cyber_score = _safe_score(cyber)
    deepfake_score = _safe_score(deepfake)

    evidence_score = _safe_score(
        evidence,
        "evidence_reliability_score"
    )

    overall_score = _calculate_overall_score(
        cyber,
        deepfake,
        selected
    )

    overall_risk = (
        "Unknown"
        if selected == ["out_of_scope"]
        else _get_risk_level(overall_score)
    )

    specialist_confidence = _get_confidence(
        cyber,
        deepfake,
        selected
    )

    overall_confidence = _combine_confidence(
        specialist_confidence,
        evidence_score
    )

    findings = _collect_findings(
        cyber,
        deepfake,
        selected
    )

    analysis = _build_overall_analysis(
        cyber,
        deepfake,
        evidence,
        overall_score,
        overall_risk,
        selected
    )

    limitations = [
        "This is an automated image-based analysis.",
        (
            "Results depend on the quality of the submitted "
            "image and available evidence."
        ),
    ]

    if set(selected) == {
        "cybersecurity",
        "deepfake",
    }:
        limitations.append(
            "The assessment covers both cybersecurity threats "
            "and deepfake or visual authenticity indicators."
        )

    elif selected == ["cybersecurity"]:
        limitations.append(
            "The assessment focuses only on cybersecurity threats."
        )

    elif selected == ["deepfake"]:
        limitations.append(
            "The assessment focuses only on deepfake "
            "and visual authenticity."
        )

    actions = []

    if (
        "cybersecurity" in selected
        and isinstance(
            cyber.get("recommended_action"),
            list
        )
    ):
        actions.extend(
            str(x).strip()
            for x in cyber["recommended_action"]
            if str(x).strip()
        )

    if (
        "deepfake" in selected
        and isinstance(
            deepfake.get("recommended_action"),
            list
        )
    ):
        actions.extend(
            str(x).strip()
            for x in deepfake["recommended_action"]
            if str(x).strip()
        )

    recommended = list(
        dict.fromkeys(actions)
    )

    if (
        not recommended
        and selected != ["out_of_scope"]
    ):
        recommended = [
            "Review the important findings manually.",
            "Verify suspicious information using reliable sources.",
        ]

    elapsed = round(
        time.perf_counter() - start,
        2
    )

    final_state = dict(state)

    final_state.update({
        "cybersecurity_risk_score": cyber_score,
        "deepfake_risk_score": deepfake_score,
        "evidence_reliability_score": evidence_score,
        "overall_threat_score": overall_score,
        "overall_risk_level": overall_risk,
        "overall_confidence": overall_confidence,
        "key_findings": findings,
        "overall_analysis": analysis,
        "limitations": limitations,
        "recommended_action": recommended,
        "routing_path": _get_routing_path(selected),
        "media_routing_path": _get_routing_path(selected),
        "media_routing_reason": state.get(
            "routing_reason",
            "Not available."
        ),
        "total_processing_time": elapsed,
    })

    return {
        **{
            k: final_state[k]
            for k in (
                "cybersecurity_risk_score",
                "deepfake_risk_score",
                "evidence_reliability_score",
                "overall_threat_score",
                "overall_risk_level",
                "overall_confidence",
                "key_findings",
                "overall_analysis",
                "limitations",
                "recommended_action",
                "routing_path",
                "media_routing_path",
                "media_routing_reason",
                "total_processing_time",
            )
        },
        "final_response": _build_final_response(
            final_state
        ),
    }


def _wrap_text(
    text: str,
    width: int = 65,
) -> list[str]:
    """Wrap long text for readable console output."""
    return textwrap.wrap(
        str(text),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _add_findings(
    lines: list[str],
    findings: list[str],
) -> None:
    """Append formatted findings."""

    if findings:
        for finding in findings:
            wrapped = _wrap_text(finding)

            if wrapped:
                lines.append(
                    f"- {wrapped[0]}"
                )

                lines.extend(
                    f"  {x}"
                    for x in wrapped[1:]
                )

    else:
        lines.append(
            "- No significant findings were identified."
        )


def _build_final_response(
    state: dict[str, Any],
) -> str:
    """Build the final readable response for all routing paths."""

    selected = list(
        state.get(
            "selected_agents",
            []
        )
    )

    cyber_score = state.get(
        "cybersecurity_risk_score",
        0
    )

    deepfake_score = state.get(
        "deepfake_risk_score",
        0
    )

    risk = state.get(
        "overall_risk_level",
        "Unknown"
    )

    confidence = state.get(
        "overall_confidence",
        "Unknown"
    )

    findings = state.get(
        "key_findings",
        []
    )

    analysis = state.get(
        "overall_analysis",
        ""
    )

    actions = state.get(
        "recommended_action",
        []
    )

    processing = state.get(
        "total_processing_time",
        0
    )

    lines = []

    # Human developer modification: keep Partial and Error warnings separate.
    if (
        state.get("fatal_error", False)
        or state.get("router_status", "") == "ERROR"
    ):
        lines += [
            "⚠ ERROR WARNING",
            "----------------------------------------------------------------------",
            "The request could not be processed.",
            "Please check the input and image path, then try again.",
            "",
        ]

    elif (
        state.get("llm_unavailable", False)
        or state.get("router_status", "") == "FALLBACK"
    ):
        lines += [
            "⚠ PARTIAL WARNING",
            "----------------------------------------------------------------------",
            "The AI service was temporarily unavailable.",
            "A partial result was generated using available detection tools.",
            "Please interpret the result with caution.",
            "",
        ]

    if state.get("fatal_error", False):
        lines += [
            "ANALYSIS RESULT",
            "----------------------------------------------------------------------",
            "No analysis result is available because processing failed.",
            "",
        ]

    elif selected == ["out_of_scope"]:
        lines += [
            "ANALYSIS RESULT",
            "----------------------------------------------------------------------",
            (
                "The request is outside the supported "
                "cybersecurity and deepfake analysis scope."
            ),
            "",
        ]

    else:
        both = set(selected) == {
            "cybersecurity",
            "deepfake",
        }

        title = (
            "CYBERSECURITY + DEEPFAKE ANALYSIS"
            if both
            else (
                "CYBERSECURITY ANALYSIS"
                if selected == ["cybersecurity"]
                else "DEEPFAKE ANALYSIS"
            )
        )

        lines += [
            "----------------------------------------------------------------------",
            title,
            "----------------------------------------------------------------------",
            "",
        ]

        if both:
            lines.append(
                f"Cybersecurity Risk       : {cyber_score}%"
            )

            lines.append(
                f"Deepfake Risk            : {deepfake_score}%"
            )

        elif selected == ["cybersecurity"]:
            lines.append(
                f"Risk Score               : {cyber_score}%"
            )

        else:
            lines.append(
                f"Deepfake Risk            : {deepfake_score}%"
            )

        lines += [
            f"Risk Level               : {risk}",
            f"Confidence               : {confidence}",
            "",
            "KEY FINDINGS",
            "----------------------------------------------------------------------",
        ]

        _add_findings(
            lines,
            findings
        )

        lines += [
            "",
            "ANALYSIS",
            "----------------------------------------------------------------------",
        ]

        lines.extend(
            _wrap_text(analysis)
        )

        lines += [
            "",
            "RECOMMENDED ACTION",
            "----------------------------------------------------------------------",
        ]

        for action in actions:
            wrapped = _wrap_text(action)

            if wrapped:
                lines.append(
                    f"- {wrapped[0]}"
                )

                lines.extend(
                    f"  {x}"
                    for x in wrapped[1:]
                )

        lines.append("")

    lines += [
        f"Processing Time         : {processing} seconds",
        "",
        "======================================================================",
    ]

    return "\n".join(lines)


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph():
    """Build and compile the LangGraph multi-agent workflow."""

    workflow = StateGraph(MASState)

    workflow.add_node(
        "workflow_execution_guard",
        workflow_execution_guard
    )

    workflow.add_node(
        "input_validation",
        input_validation_node
    )

    workflow.add_node(
        "error_response",
        error_response_node
    )

    workflow.add_node(
        "fallback_router",
        fallback_router_node
    )

    workflow.add_node(
        "ocr",
        ocr_node
    )

    workflow.add_node(
        "router",
        router_node
    )

    workflow.add_node(
        "cybersecurity",
        cybersecurity_node
    )

    workflow.add_node(
        "deepfake",
        deepfake_node
    )

    workflow.add_node(
        "evidence",
        evidence_node
    )

    workflow.add_node(
        "final_response",
        final_response_node
    )

    workflow.add_conditional_edges(
        START,
        route_after_execution_guard,
        {
            "continue": "workflow_execution_guard",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "workflow_execution_guard",
        route_after_execution_guard,
        {
            "continue": "input_validation",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "input_validation",
        route_after_input_validation,
        {
            "ocr": "ocr",
            "fallback": "fallback_router",
            "error": "error_response",
        },
    )

    workflow.add_conditional_edges(
        "ocr",
        route_after_ocr,
        {
            "router": "router",
            "error": "error_response",
        },
    )

    # Human developer modification: explicitly map every Dynamic Router
    # output to the correct workflow node so LangGraph does not infer
    # unknown routes as END.
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "cybersecurity": "cybersecurity",
            "deepfake": "deepfake",
            "final_response": "final_response",
        },
    )

    # Human developer modification: the deterministic fallback Router
    # uses the same specialist routing paths without an LLM call.
    workflow.add_conditional_edges(
        "fallback_router",
        route_after_router,
        {
            "cybersecurity": "cybersecurity",
            "deepfake": "deepfake",
            "final_response": "final_response",
        },
    )

    workflow.add_edge(
        "error_response",
        "final_response"
    )

    workflow.add_edge(
        "cybersecurity",
        "evidence"
    )

    workflow.add_edge(
        "deepfake",
        "evidence"
    )

    workflow.add_edge(
        "evidence",
        "final_response"
    )

    workflow.add_edge(
        "final_response",
        END
    )

    return workflow.compile()