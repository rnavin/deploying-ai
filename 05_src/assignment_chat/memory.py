from typing import Dict, Any, List, Tuple

class MemoryManager:
    """
    Simple short-term memory:
    - keep last N turns verbatim
    - summarize older turns when length grows
    """
    def __init__(self, llm, max_turns_before_summarize: int = 14, keep_last_turns: int = 8):
        self.llm = llm
        self.max_turns_before_summarize = max_turns_before_summarize
        self.keep_last_turns = keep_last_turns

    def reset(self) -> Dict[str, Any]:
        return {"turns": [], "summary": ""}

    def add_turn(self, mem_state: Dict[str, Any], user: str, assistant: str) -> Dict[str, Any]:
        turns: List[Tuple[str, str]] = mem_state.get("turns") or []
        summary: str = mem_state.get("summary") or ""

        turns.append((user, assistant))

        # Summarize if too many turns
        if len(turns) > self.max_turns_before_summarize:
            old_turns = turns[:-self.keep_last_turns]
            keep_turns = turns[-self.keep_last_turns:]

            old_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in old_turns]).strip()
            new_summary = self.llm.summarize_memory(summary, old_text)

            mem_state["summary"] = new_summary
            mem_state["turns"] = keep_turns
            return mem_state

        mem_state["turns"] = turns
        mem_state["summary"] = summary
        return mem_state