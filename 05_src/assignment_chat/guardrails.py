import re
from typing import Tuple

# Restricted topics per assignment
RESTRICTED_PATTERNS = [
    r"\bcats?\b", r"\bdogs?\b",
    r"\bhoroscope(s)?\b", r"\bzodiac\b", r"\bzodiac sign(s)?\b",
    r"\btaylor swift\b",
]

# Attempts to reveal or modify system prompt
PROMPT_ATTACK_PATTERNS = [
    r"system prompt", r"reveal.*prompt", r"show.*prompt",
    r"ignore.*previous", r"forget.*instructions",
    r"you are now", r"change.*system", r"modify.*system",
]

# High-risk medical situations (dietitian bot should refuse)
MEDICAL_HIGH_RISK_PATTERNS = [
    r"\bchest pain\b",
    r"\btrouble breathing\b|\bshortness of breath\b",
    r"\bfainting\b|\bpassed out\b",
    r"\bstroke\b|\bface droop\b|\bslurred speech\b",
    r"\bsuicid(al)?\b|\bself-harm\b",
    r"\banorexia\b|\bbulimia\b|\beating disorder\b",
    r"\bpregnan(t|cy)\b|\bpostpartum\b",
    r"\binsulin\b|\bwarfarin\b|\bblood thinner\b",
]

def guardrails_check(user_text: str) -> Tuple[bool, str]:
    """
    Returns: (blocked: bool, refusal_message: str)
    """
    txt = (user_text or "").lower()

    # Block prompt attacks
    if any(re.search(p, txt) for p in PROMPT_ATTACK_PATTERNS):
        return True, "Nice try 🙂 I can’t reveal or modify my system instructions. Ask me a normal nutrition question instead."

    # Block restricted topics
    if any(re.search(p, txt) for p in RESTRICTED_PATTERNS):
        return True, "I can’t help with that topic. Ask me something else (nutrition, meal planning, habits, etc.)."

    # Medical high-risk refusal
    if any(re.search(p, txt) for p in MEDICAL_HIGH_RISK_PATTERNS):
        return True, (
            "I can’t safely help with this situation as a nutrition coach. "
            "Please contact a licensed clinician or local emergency services if this is urgent. "
            "If you want, tell me your general nutrition goal and preferences and I’ll offer safe general guidance."
        )

    return False, ""