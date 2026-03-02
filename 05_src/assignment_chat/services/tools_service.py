import re
from typing import Any, Dict, Optional

class ToolsOrchestrator:
    """
    Simple tools service:
    - convert: basic unit conversions (km<->miles)
    - calculate: safe arithmetic with a whitelist
    You can expand later using OpenAI function calling if you want.
    """

    def __init__(self, llm=None):
        # llm is optional; kept for compatibility with app.py and future upgrades
        self.llm = llm

    # ----------------------------
    # Public entry point
    # ----------------------------
    def run(self, user_message: str, mem_state: Optional[Dict[str, Any]] = None) -> str:
        txt = (user_message or "").strip()

        # convert examples: "convert 10 km to miles"
        if re.search(r"^\s*convert\b", txt, flags=re.I):
            out = self._handle_convert(txt)
            if out:
                return out
            return "I can convert simple units like `convert 10 km to miles`."

        # calculate examples: "calculate (12.5*8)/2"
        if re.search(r"^\s*(calculate|calc)\b", txt, flags=re.I):
            expr = re.sub(r"^\s*(calculate|calc)\b", "", txt, flags=re.I).strip()
            out = self._safe_eval(expr)
            if out is None:
                return "I can only calculate basic arithmetic (+ - * / parentheses). Example: `calculate (12.5*8)/2`."
            return f"Result: {out}"

        return "Tools: try `convert 10 km to miles` or `calculate (12.5*8)/2`."

    # ----------------------------
    # Conversion helper
    # ----------------------------
    def _handle_convert(self, txt: str) -> Optional[str]:
        m = re.search(
            r"convert\s+([0-9]*\.?[0-9]+)\s*(km|kilometer|kilometers|mi|mile|miles)\s+to\s+(km|kilometer|kilometers|mi|mile|miles)",
            txt,
            flags=re.I,
        )
        if not m:
            return None

        value = float(m.group(1))
        src = m.group(2).lower()
        dst = m.group(3).lower()

        km_alias = {"km", "kilometer", "kilometers"}
        mi_alias = {"mi", "mile", "miles"}

        if src in km_alias and dst in mi_alias:
            miles = value * 0.621371
            return f"{value:g} km ≈ {miles:.4g} miles"
        if src in mi_alias and dst in km_alias:
            km = value / 0.621371
            return f"{value:g} miles ≈ {km:.4g} km"

        return None

    # Safe calculator (no abuse)
   
    def _safe_eval(self, expr: str) -> Optional[float]:
        expr = (expr or "").strip()
        if not expr:
            return None

        # allow only digits, operators, parentheses, decimal points, and whitespace
        if re.search(r"[^0-9\.\+\-\*\/\(\)\s]", expr):
            return None

        try:
            # eval with empty builtins
            val = eval(expr, {"__builtins__": {}}, {})
        except Exception:
            return None

        if isinstance(val, (int, float)):
            return float(val)

        return None