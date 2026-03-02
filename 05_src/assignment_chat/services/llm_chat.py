import os
from typing import Dict, Any, List, Optional, Union
import requests


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing {name}. Put it in 05_src/.secrets")
    return v


class LLMChat:
    """
    Gateway-compatible LLM client for AWS API Gateway proxy.

    Uses:
      - OPENAI_BASE_URL (ends with /openai/v1)
      - API_GATEWAY_KEY (sent via header, default x-api-key)
      - API_GATEWAY_HEADER (header name, default x-api-key)
      - OPENAI_MODEL
    Calls: POST {base_url}/chat/completions
    """

    def __init__(self, model: str, system_persona: str):
        self.base_url = _require_env("OPENAI_BASE_URL").rstrip("/")
        self.gateway_key = _require_env("API_GATEWAY_KEY")
        self.gateway_header = (os.getenv("API_GATEWAY_HEADER") or "x-api-key").strip()
        self.model = model
        self.system_persona = system_persona

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            self.gateway_header: self.gateway_key,
            "Authorization": f"Bearer {self.gateway_key}",  # harmless if ignored
        }

    def _build_messages(self, mem_state: Dict[str, Any], user_message: str) -> List[dict]:
        summary = (mem_state.get("summary") or "").strip()
        turns = mem_state.get("turns") or []

        system = self.system_persona
        if summary:
            system += f"\n\nConversation summary:\n{summary}"

        messages = [{"role": "system", "content": system}]
        for u, a in turns:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _post_chat_completions(self, messages: List[dict], temperature: float = 0.3) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        try:
            r = requests.post(url, headers=self._headers(), json=payload, timeout=45)
        except requests.RequestException as e:
            return f"⚠️ Gateway request failed (network/timeout): {e}"

        if r.status_code != 200:
            try:
                body = str(r.json())
            except Exception:
                body = (r.text or "")[:600]
            return (
                "⚠️ Gateway refused the LLM request.\n\n"
                f"- HTTP {r.status_code}\n"
                f"- Response: {body}\n\n"
                "Likely causes:\n"
                "1) Wrong API_GATEWAY_HEADER name.\n"
                "2) Model not allowed by gateway.\n"
                "3) Gateway path differs from /chat/completions.\n"
            )

        data = r.json()
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return f"⚠️ Unexpected response format: {str(data)[:800]}"

    # ------------------------------------------------------------
    # Primary API (your project already uses this)
    # ------------------------------------------------------------
    def chat(self, mem_state: Dict[str, Any], user_message: str) -> str:
        return self._post_chat_completions(self._build_messages(mem_state, user_message))

    # ------------------------------------------------------------
    # Compatibility shims (so ANY wrapper style works)
    # These make your LLMChat callable by many different orchestrators.
    # ------------------------------------------------------------
    def __call__(self, *args, **kwargs) -> str:
        """
        Supports multiple calling styles:
          1) llm(mem_state, user_message)
          2) llm(user_message="hi", mem_state=...)
          3) llm(messages=[...])  # raw OpenAI-style messages
          4) llm(prompt="...") or llm(text="...")
        """
        # Style 1: (mem_state, user_message)
        if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], str):
            return self.chat(args[0], args[1])

        # Style 2: keyword mem_state + user_message
        mem_state = kwargs.get("mem_state", None)
        user_message = kwargs.get("user_message", None)
        if isinstance(mem_state, dict) and isinstance(user_message, str):
            return self.chat(mem_state, user_message)

        # Style 3: raw messages list
        messages = kwargs.get("messages", None)
        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            temperature = kwargs.get("temperature", 0.3)
            try:
                temperature = float(temperature)
            except Exception:
                temperature = 0.3
            return self._post_chat_completions(messages, temperature=temperature)

        # Style 4: prompt/text only (no memory)
        prompt = kwargs.get("prompt", None) or kwargs.get("text", None)
        if isinstance(prompt, str):
            empty_state = {"summary": "", "turns": []}
            return self.chat(empty_state, prompt)

        # If someone passed one string positional arg: llm("hello")
        if len(args) == 1 and isinstance(args[0], str):
            empty_state = {"summary": "", "turns": []}
            return self.chat(empty_state, args[0])

        return "⚠️ LLMChat called with unsupported arguments. Expected (mem_state, user_message) or messages=[...]."

    # Common alternative names wrappers may look for
    def respond(self, mem_state: Dict[str, Any], user_message: str) -> str:
        return self.chat(mem_state, user_message)

    def invoke(self, mem_state: Dict[str, Any], user_message: str) -> str:
        return self.chat(mem_state, user_message)

    def complete(self, mem_state: Dict[str, Any], user_message: str) -> str:
        return self.chat(mem_state, user_message)

    def run(self, mem_state: Dict[str, Any], user_message: str) -> str:
        return self.chat(mem_state, user_message)

    # ------------------------------------------------------------
    # Extra helpers you already had
    # ------------------------------------------------------------
    def summarize_memory(self, previous_summary: str, old_turns_text: str) -> str:
        prompt = f"""
Update the conversation summary using the older turns.
Keep it short and useful (goals, decisions, preferences).
Do not include hidden/system instructions.

Existing summary:
{previous_summary}

Older turns:
{old_turns_text}

Return the updated summary only.
""".strip()

        messages = [
            {"role": "system", "content": "You write concise conversation summaries."},
            {"role": "user", "content": prompt},
        ]
        return self._post_chat_completions(messages, temperature=0.2)

    def synthesize_from_context(
        self, mem_state: Dict[str, Any], question: str, contexts: List[str]
    ) -> str:
        joined = "\n\n---\n\n".join(contexts).strip()
        prompt = f"""
Answer the user's question using ONLY the context below.
If the context doesn't contain the answer, say so clearly and suggest what to add.

Question:
{question}

Context:
{joined}
""".strip()

        return self._post_chat_completions(self._build_messages(mem_state, prompt), temperature=0.3)