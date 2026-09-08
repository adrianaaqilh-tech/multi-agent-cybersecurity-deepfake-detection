"""Tools for preliminary evidence consistency analysis."""


def evidence_consistency_checker(
    cybersecurity_result: dict | None,
    deepfake_result: dict | None,
) -> dict:
    """
    Evaluate the consistency and reliability of specialist findings.

    The tool supports cybersecurity-only, deepfake-only, and
    combined cybersecurity + deepfake analysis.

    Args:
        cybersecurity_result:
            Result from the Cybersecurity Agent. Empty when the
            Cybersecurity Agent was not selected.

        deepfake_result:
            Result from the Deepfake Agent. Empty when the
            Deepfake Agent was not selected.

    Returns:
        A dictionary containing specialist scores, findings,
        consistency, support level, reliability score, and issues.
    """

    # Human developer modification:
    # Use empty dictionaries when a specialist was not selected.
    cybersecurity_result = cybersecurity_result or {}
    deepfake_result = deepfake_result or {}

    # ============================================================
    # DETERMINE ACTIVE SPECIALISTS
    # ============================================================

    cybersecurity_selected = bool(
        cybersecurity_result
    )

    deepfake_selected = bool(
        deepfake_result
    )

    # ============================================================
    # EXTRACT CYBERSECURITY EVIDENCE
    # ============================================================

    cybersecurity_score = float(
        cybersecurity_result.get(
            "risk_score",
            0,
        )
    )

    cybersecurity_findings = cybersecurity_result.get(
        "findings",
        [],
    )

    # ============================================================
    # EXTRACT DEEPFAKE EVIDENCE
    # ============================================================

    deepfake_score = float(
        deepfake_result.get(
            "risk_score",
            0,
        )
    )

    deepfake_findings = deepfake_result.get(
        "findings",
        [],
    )

    # ============================================================
    # DETERMINE SELECTED AGENT
    # ============================================================

    if (
        cybersecurity_selected
        and deepfake_selected
    ):

        selected_agent = (
            "Cybersecurity Threat Detection Agent "
            "+ Deepfake Detection Agent"
        )

    elif cybersecurity_selected:

        selected_agent = (
            "Cybersecurity Threat Detection Agent"
        )

    elif deepfake_selected:

        selected_agent = (
            "Deepfake Detection Agent"
        )

    else:

        selected_agent = "Unknown"

    # ============================================================
    # SELECT PRIMARY EVIDENCE
    # ============================================================

    if (
        cybersecurity_selected
        and deepfake_selected
    ):

        # Human developer modification:
        # When both specialists run, combine findings from both
        # agents instead of treating one agent as the selected result.
        selected_score = max(
            cybersecurity_score,
            deepfake_score,
        )

        selected_findings = (
            cybersecurity_findings
            + deepfake_findings
        )

    elif cybersecurity_selected:

        selected_score = cybersecurity_score

        selected_findings = (
            cybersecurity_findings
        )

    elif deepfake_selected:

        selected_score = deepfake_score

        selected_findings = (
            deepfake_findings
        )

    else:

        selected_score = 0.0

        selected_findings = []

    # ============================================================
    # CHECK AVAILABLE EVIDENCE
    # ============================================================

    issues = []

    if not cybersecurity_selected and not deepfake_selected:

        issues.append(
            "No specialist finding is available for verification."
        )

    if cybersecurity_selected and not cybersecurity_findings:

        issues.append(
            "Cybersecurity Agent returned no specific findings."
        )

    if deepfake_selected and not deepfake_findings:

        issues.append(
            "Deepfake Agent returned no specific findings."
        )

    # ============================================================
    # EVIDENCE CONSISTENCY
    # ============================================================

    if not cybersecurity_selected and not deepfake_selected:

        consistency = "Insufficient"

    elif (
        cybersecurity_selected
        and deepfake_selected
    ):

        if (
            cybersecurity_findings
            and deepfake_findings
        ):
            consistency = "High"

        elif (
            cybersecurity_findings
            or deepfake_findings
        ):
            consistency = "Partial"

        else:
            consistency = "Low"

    elif selected_findings:

        consistency = "High"

    else:

        consistency = "Low"

    # ============================================================
    # SUPPORT LEVEL
    # ============================================================

    if not cybersecurity_selected and not deepfake_selected:

        support_level = "Insufficient"

    elif (
        cybersecurity_selected
        and deepfake_selected
    ):

        if (
            cybersecurity_findings
            and deepfake_findings
        ):
            support_level = "Strong"

        elif (
            cybersecurity_findings
            or deepfake_findings
        ):
            support_level = "Partial"

        else:
            support_level = "Insufficient"

    elif selected_findings:

        support_level = "Strong"

    else:

        support_level = "Insufficient"

    # ============================================================
    # RELIABILITY SCORE
    # ============================================================

    if not cybersecurity_selected and not deepfake_selected:

        reliability_score = 20

    elif (
        cybersecurity_selected
        and deepfake_selected
        and cybersecurity_findings
        and deepfake_findings
    ):

        # Human developer modification:
        # Evidence from both specialist agents provides stronger
        # cross-agent support when both produce findings.
        reliability_score = 90

    elif (
        cybersecurity_selected
        and deepfake_selected
    ):

        reliability_score = 60

    elif selected_findings:

        reliability_score = 80

    else:

        reliability_score = 30

    # ============================================================
    # SCORE-BASED EVIDENCE CHECK
    # ============================================================

    # A low score with findings means evidence exists but does not
    # strongly support a high-risk conclusion.
    if (
        selected_findings
        and selected_score < 40
    ):

        if reliability_score > 60:
            reliability_score = 60

        if support_level == "Strong":
            support_level = "Partial"

        if consistency == "High":
            consistency = "Partial"

    # ============================================================
    # RETURN RESULT
    # ============================================================

    return {
        "selected_agent":
            selected_agent,

        "selected_score":
            selected_score,

        "selected_findings":
            selected_findings,

        "cybersecurity_score":
            cybersecurity_score,

        "deepfake_score":
            deepfake_score,

        "cybersecurity_findings":
            cybersecurity_findings,

        "deepfake_findings":
            deepfake_findings,

        "agent_consistency":
            consistency,

        "support_level":
            support_level,

        "reliability_score":
            reliability_score,

        "issues":
            issues,
    }