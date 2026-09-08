"""Dynamic Router for the Cybersecurity and Deepfake Multi-Agent System."""

import json

from config import llm, is_llm_available


# ============================================================
# DYNAMIC ROUTER PROMPT
# ============================================================

DYNAMIC_ROUTER_PROMPT = """
You are the Dynamic Router for a Multi-Agent System that detects
cybersecurity threats and deepfake/image authenticity issues.

Your task is to decide which specialist agent(s) should analyse
the user's request.

IMPORTANT:
- The USER REQUEST is the PRIMARY signal.
- OCR text is only supporting information.
- Do not let OCR override the user's requested task.
- Use the meaning of the request, not only exact keywords.

AVAILABLE ROUTES:

1. cybersecurity
Use for phishing, scams, suspicious links, malicious URLs,
fraud, malware, hacking, cyber attacks, account security,
online security, or other cybersecurity threats.

2. deepfake
Use for deepfake detection, AI-generated images/media,
image authenticity, image manipulation, fake images,
edited faces, or digitally altered images.

3. both
Use when the user requests BOTH cybersecurity analysis
AND deepfake/image-authenticity analysis.

4. out_of_scope
Use when the request is unrelated to cybersecurity
or deepfake/image-authenticity analysis.

FEW-SHOT EXAMPLES:

Example 1:
User: "Please check this image for phishing."
Answer:
{"selected_agents":["cybersecurity"],"routing_reason":"The user requests cybersecurity analysis."}

Example 2:
User: "Can you check whether this image is AI generated?"
Answer:
{"selected_agents":["deepfake"],"routing_reason":"The user requests image authenticity analysis."}

Example 3:
User: "Please check this image for phishing and determine if it is AI generated."
Answer:
{"selected_agents":["cybersecurity","deepfake"],"routing_reason":"The user requests both cybersecurity and deepfake analysis."}

Example 4:
User: "Can you give me a cooking recipe?"
Answer:
{"selected_agents":["out_of_scope"],"routing_reason":"The request is unrelated to cybersecurity or deepfake analysis."}

OUTPUT RULES:
- Return ONLY one valid JSON object.
- Do not use Markdown.
- Do not add explanations outside the JSON.
- selected_agents must contain:
  ["cybersecurity"]
  OR ["deepfake"]
  OR ["cybersecurity","deepfake"]
  OR ["out_of_scope"]
- Never return "both" as an agent.
- Never select the Evidence Agent.
"""


# ============================================================
# VALID AGENTS
# ============================================================

VALID_AGENTS = {
    "cybersecurity",
    "deepfake",
    "out_of_scope",
}


# ============================================================
# VALIDATION
# ============================================================

def _validate_routing_result(result: dict) -> dict:
    """
    Validate the routing decision returned by the LLM.

    Args:
        result: Parsed JSON result from the Dynamic Router.

    Returns:
        Validated routing dictionary.

    Raises:
        ValueError: If the routing result is invalid.
    """

    if not isinstance(result, dict):
        raise ValueError(
            "Router response must be a JSON object."
        )

    if "selected_agents" not in result:
        raise ValueError(
            "Router response is missing selected_agents."
        )

    if "routing_reason" not in result:
        raise ValueError(
            "Router response is missing routing_reason."
        )

    selected_agents = result["selected_agents"]

    if not isinstance(selected_agents, list):
        raise ValueError(
            "selected_agents must be a list."
        )

    if len(selected_agents) == 0:
        raise ValueError(
            "Router must select at least one routing option."
        )

    if len(selected_agents) > 2:
        raise ValueError(
            "Router cannot select more than two specialist agents."
        )

    # --------------------------------------------------------
    # AGENT VALIDATION
    # --------------------------------------------------------

    for agent in selected_agents:
        if agent not in VALID_AGENTS:
            raise ValueError(
                f"Invalid routing option: {agent}"
            )

    # --------------------------------------------------------
    # OUT-OF-SCOPE VALIDATION
    # --------------------------------------------------------

    if "out_of_scope" in selected_agents:
        if len(selected_agents) != 1:
            raise ValueError(
                "out_of_scope cannot be combined with specialist agents."
            )

    # --------------------------------------------------------
    # DUPLICATE VALIDATION
    # --------------------------------------------------------

    if len(selected_agents) != len(set(selected_agents)):
        raise ValueError(
            "Duplicate routing agents are not allowed."
        )

    # --------------------------------------------------------
    # BOTH SPECIALIST VALIDATION
    # --------------------------------------------------------

    if (
        "cybersecurity" in selected_agents
        and "deepfake" in selected_agents
    ):
        selected_agents = [
            "cybersecurity",
            "deepfake",
        ]

    elif selected_agents[0] in {
        "cybersecurity",
        "deepfake",
    }:
        selected_agents = [
            selected_agents[0]
        ]

    routing_reason = str(
        result["routing_reason"]
    ).strip()

    if not routing_reason:
        routing_reason = "Routing decision completed."

    return {
        "selected_agents": selected_agents,
        "routing_reason": routing_reason,
    }


# ============================================================
# USER-INTENT FALLBACK
# ============================================================

def _keyword_route(user_request: str) -> dict:
    """
    Perform deterministic fallback routing when the Dynamic Router fails.

    The fallback uses only the user's request and never uses OCR.

    Args:
        user_request: Original user request.

    Returns:
        Deterministic routing result.
    """

    request = (
        user_request or ""
    ).lower().strip()

    # --------------------------------------------------------
    # CYBERSECURITY KEYWORDS
    # --------------------------------------------------------

    cybersecurity_keywords = [
        "cybersecurity",
        "cyber security",
        "cybersecurity threat",
        "cyber threat",
        "cyber attack",
        "cyber attack",
        "phishing",
        "phish",
        "scam",
        "fraud",
        "suspicious url",
        "suspicious link",
        "malicious link",
        "malware",
        "ransomware",
        "hacking",
        "hack",
        "social engineering",
        "account security",
        "account verification",
        "online security",
        "security threat",
        "fraudulent message",
        "banking scam",
        "bank scam",
        "fake bank message",
        "suspicious message",
    ]

    # --------------------------------------------------------
    # DEEPFAKE KEYWORDS
    # --------------------------------------------------------

    deepfake_keywords = [
        "deepfake",
        "deep fake",
        "ai generated",
        "ai-generated",
        "ai generated image",
        "ai-generated image",
        "ai generated face",
        "ai-generated face",
        "ai generated media",
        "ai-generated media",
        "image manipulation",
        "image manipulated",
        "visual authenticity",
        "face manipulation",
        "identity manipulation",
        "digitally altered",
        "edited face",
        "fake image",
        "image authenticity",
        "image is fake",
        "ai image",
        "generated image",
        "manipulated image",
    ]

    cybersecurity_requested = any(
        keyword in request
        for keyword in cybersecurity_keywords
    )

    deepfake_requested = any(
        keyword in request
        for keyword in deepfake_keywords
    )

    # --------------------------------------------------------
    # BOTH
    # --------------------------------------------------------

    # Human developer modification:
    # When both cybersecurity and deepfake intent are detected,
    # both specialist agents are selected.

    if (
        cybersecurity_requested
        and deepfake_requested
    ):
        return {
            "selected_agents": [
                "cybersecurity",
                "deepfake",
            ],
            "routing_reason": (
                "The user request contains both cybersecurity "
                "and deepfake or image-authenticity intent."
            ),
        }

    # --------------------------------------------------------
    # CYBERSECURITY ONLY
    # --------------------------------------------------------

    if cybersecurity_requested:
        return {
            "selected_agents": [
                "cybersecurity",
            ],
            "routing_reason": (
                "The user request specifically asks "
                "for cybersecurity analysis."
            ),
        }

    # --------------------------------------------------------
    # DEEPFAKE ONLY
    # --------------------------------------------------------

    if deepfake_requested:
        return {
            "selected_agents": [
                "deepfake",
            ],
            "routing_reason": (
                "The user request specifically asks "
                "for deepfake or image-authenticity analysis."
            ),
        }

    # --------------------------------------------------------
    # OUT OF SCOPE
    # --------------------------------------------------------

    return {
        "selected_agents": [
            "out_of_scope",
        ],
        "routing_reason": (
            "The request does not clearly relate to "
            "cybersecurity or deepfake detection."
        ),
    }


# ============================================================
# DYNAMIC ROUTING
# ============================================================

def _dynamic_route(
    user_request: str,
    image_text: str,
) -> dict:
    """
    Use Gemma to dynamically determine the routing decision.

    The user's request is the primary signal and OCR is secondary.

    Args:
        user_request: Original user request.
        image_text: OCR text extracted from the image.

    Returns:
        Validated Dynamic Router result.
    """

    # Human developer modification:
    # Limit OCR because OCR is only supporting context and
    # should not make the router prompt unnecessarily large.
    max_ocr_chars = 800

    limited_ocr = (
        image_text or ""
    ).strip()[:max_ocr_chars]

    if not limited_ocr:
        limited_ocr = (
            "No readable OCR text was detected."
        )

    # Human developer modification:
    # Keep the user request clearly separated from OCR so
    # Gemma prioritises the user's actual requested task.
    prompt = f"""
{DYNAMIC_ROUTER_PROMPT}

USER REQUEST:
{user_request}

OCR SUPPORTING TEXT:
{limited_ocr}

Now classify the USER REQUEST.

Return ONLY the JSON object.
"""

    response = llm.invoke(prompt)

    content = response.content

    # --------------------------------------------------------
    # HANDLE DIFFERENT RESPONSE FORMATS
    # --------------------------------------------------------

    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    content = str(content).strip()

    # --------------------------------------------------------
    # REMOVE MARKDOWN CODE FENCES
    # --------------------------------------------------------

    # Human developer modification:
    # Gemma may wrap JSON inside Markdown code fences.
    # Removing them allows valid JSON responses to be accepted.

    if content.startswith("```json"):
        content = content[7:]

    elif content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    # --------------------------------------------------------
    # EXTRACT JSON OBJECT
    # --------------------------------------------------------

    # Human developer modification:
    # Extract only the JSON object so accidental text from
    # Gemma does not cause the entire routing step to fail.

    json_start = content.find("{")
    json_end = content.rfind("}")

    if (
        json_start == -1
        or json_end == -1
        or json_end <= json_start
    ):
        raise ValueError(
            "Dynamic Router did not return a valid JSON object."
        )

    content = content[
        json_start:json_end + 1
    ]

    parsed_result = json.loads(
        content
    )

    return _validate_routing_result(
        parsed_result
    )


# ============================================================
# DYNAMIC ROUTER AGENT
# ============================================================

def router_agent(
    user_request: str,
    image_text: str,
) -> dict:
    """
    Run the Dynamic Router silently.

    LM Studio is checked before the LLM call. If the service
    is available, Gemma performs semantic routing.

    If the LLM call genuinely fails, deterministic routing based
    only on the user's request is used.

    Args:
        user_request: Original user request.
        image_text: OCR text extracted from the image.

    Returns:
        Routing decision containing one or both specialist agents,
        or an out-of-scope decision.
    """

    # ========================================================
    # CHECK LLM AVAILABILITY
    # ========================================================

    try:

        # Human developer modification:
        # Check LM Studio before making the main LLM request
        # so an offline service can quickly use the fallback.

        llm_available = is_llm_available()

    except Exception:

        llm_available = False

    if not llm_available:

        fallback_result = _keyword_route(
            user_request
        )

        return {
            "selected_agents":
                fallback_result["selected_agents"],

            "routing_reason": (
                fallback_result["routing_reason"]
                + " The LLM service is unavailable, "
                "so deterministic fallback routing was used."
            ),

            "router_status":
                "FALLBACK",
        }

    # ========================================================
    # RUN GEMMA DYNAMIC ROUTER
    # ========================================================

    try:

        # Human developer modification:
        # Gemma remains the main routing mechanism when the
        # service is available. Keyword routing is only fallback.

        routing_result = _dynamic_route(
            user_request,
            image_text,
        )

        return {
            "selected_agents":
                routing_result["selected_agents"],

            "routing_reason":
                routing_result["routing_reason"],

            "router_status":
                "SUCCESS",
        }

    # ========================================================
    # INVALID LLM RESPONSE
    # ========================================================

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):

        # Human developer modification:
        # Invalid JSON is handled safely without making another
        # LLM call. Deterministic routing prevents system failure.

        fallback_result = _keyword_route(
            user_request
        )

        return {
            "selected_agents":
                fallback_result["selected_agents"],

            "routing_reason": (
                fallback_result["routing_reason"]
                + " The Dynamic Router returned an invalid "
                "response, so deterministic routing was used."
            ),

            "router_status":
                "FALLBACK",
        }

    # ========================================================
    # UNEXPECTED LLM FAILURE
    # ========================================================

    except Exception:

        # Human developer modification:
        # Unexpected LLM failures are caught so the complete
        # Multi-Agent System does not crash.

        fallback_result = _keyword_route(
            user_request
        )

        return {
            "selected_agents":
                fallback_result["selected_agents"],

            "routing_reason": (
                fallback_result["routing_reason"]
                + " The Dynamic Router could not complete "
                "the AI routing step, so deterministic routing was used."
            ),

            "router_status":
                "FALLBACK",
        }