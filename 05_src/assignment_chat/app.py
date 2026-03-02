import os
import inspect

# ---------------------------
# Fix: avoid proxying localhost (prevents share=True forced)
# ---------------------------
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")
os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])

# ---------------------------
# Fix: gradio_client JSON schema bug (schema can be bool True/False)
# Must run BEFORE importing gradio.
# This patches BOTH get_type() and any direct schema.get(...) usage.
# ---------------------------
def _patch_gradio_client_schema_bug():
    try:
        import gradio_client.utils as gcu

        # Patch get_type(schema)
        old_get_type = gcu.get_type

        def safe_get_type(schema):
            if schema is True or schema is False or schema is None:
                return {}
            if not isinstance(schema, dict):
                return {}
            return old_get_type(schema)

        gcu.get_type = safe_get_type

        # Patch internal converter to be resilient when schema is bool
        old_json_schema_to_python_type = gcu._json_schema_to_python_type

        def safe_json_schema_to_python_type(schema, defs=None):
            if schema is True or schema is False or schema is None:
                return "Any"
            if not isinstance(schema, dict):
                return "Any"
            return old_json_schema_to_python_type(schema, defs)

        gcu._json_schema_to_python_type = safe_json_schema_to_python_type

    except Exception:
        # If patching fails, app may still run; worst case you see original error.
        pass


_patch_gradio_client_schema_bug()

import gradio as gr

from utils_secrets import apply_secrets_to_env
from services.llm_chat import LLMChat
from services.api_service import wiki_summary_service
from services.semantic_service import SemanticService
from memory import MemoryManager


APP_TITLE = "Dietitian Chat Bot (TXT/RAG + Wikipedia)"

SYSTEM_PERSONA = """
You are a helpful dietitian assistant.
Be practical, concise, and safety-minded.
If the user asks for medical advice, give general guidance and suggest consulting a professional.
""".strip()


def _env_bool(name: str, default: bool = False) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def _new_mem_state() -> dict:
    return {"summary": "", "turns": []}


def _memory_init_state(memory: MemoryManager) -> dict:
    """
    Your MemoryManager doesn't have init_state().
    So we try common names; otherwise return a simple dict.
    """
    for attr in ("init_state", "new_state", "create_state", "init"):
        if hasattr(memory, attr) and callable(getattr(memory, attr)):
            try:
                return getattr(memory, attr)()
            except TypeError:
                pass
    return _new_mem_state()


def _update_memory(memory: MemoryManager, mem_state: dict, user_msg: str, assistant_msg: str) -> dict:
    """
    Try common update methods; otherwise append turns locally.
    """
    for attr in ("update", "update_state", "add_turn", "push_turn"):
        if hasattr(memory, attr) and callable(getattr(memory, attr)):
            try:
                return getattr(memory, attr)(mem_state, user_msg, assistant_msg)
            except TypeError:
                break

    turns = mem_state.get("turns") or []
    turns.append((user_msg, assistant_msg))
    mem_state["turns"] = turns
    return mem_state


def _coerce_to_text(x) -> str:
    """
    SemanticService might return:
      - str
      - dict {answer: "..."} or {text: "..."} or {result: "..."}
      - list of docs
    Convert to a safe display string.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ("answer", "text", "result", "output", "content", "response"):
            val = x.get(key)
            if isinstance(val, str) and val.strip():
                return val
        # fallback: stringify dict
        return str(x)
    if isinstance(x, list):
        # join short list items
        try:
            return "\n".join(str(i) for i in x[:10])
        except Exception:
            return str(x)
    return str(x)


def _semantic_answer(semantic: SemanticService, mem_state: dict, question: str, k_default: int = 4) -> str:
    """
    Robust call wrapper for SemanticService.answer().

    Fixes: 'dict' object has no attribute 'strip'
    which happens when SemanticService.answer() expects question first, but gets mem_state.
    """
    question = (question or "").strip()
    if not question:
        return ""

    # Decide which top-k kwarg name is supported
    kwargs = {}
    try:
        sig = inspect.signature(semantic.answer)
        params = sig.parameters
        if "k" in params:
            kwargs["k"] = k_default
        elif "top_k" in params:
            kwargs["top_k"] = k_default
        elif "n_results" in params:
            kwargs["n_results"] = k_default
        elif "num_results" in params:
            kwargs["num_results"] = k_default
    except Exception:
        pass

    # Try likely call patterns FIRST: question must be first
    call_attempts = [
        ("q, state, kwargs", lambda: semantic.answer(question, mem_state, **kwargs)),
        ("q, kwargs",        lambda: semantic.answer(question, **kwargs)),
        ("q only",           lambda: semantic.answer(question)),
        ("state, q, kwargs", lambda: semantic.answer(mem_state, question, **kwargs)),
        ("state, q",         lambda: semantic.answer(mem_state, question)),
    ]

    last_err = None
    for _, fn in call_attempts:
        try:
            out = fn()
            return _coerce_to_text(out)
        except (AttributeError, TypeError) as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

    return f"⚠️ SemanticService error: {last_err}"


def build_app():
    # Load secrets from 05_src/.secrets
    this_dir = os.path.dirname(__file__)
    secrets_path = os.path.abspath(os.path.join(this_dir, "..", ".secrets"))
    apply_secrets_to_env(secrets_path)

    model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

    llm = LLMChat(model=model, system_persona=SYSTEM_PERSONA)
    memory = MemoryManager(llm=llm)

    # IMPORTANT: Your chroma store has ONLY assignment_kb with count=20
    # so set collection_name to assignment_kb (not assignment_chat)
    persist_dir = os.path.join(this_dir, "data", "chroma_store")
    semantic = SemanticService(
        persist_dir=persist_dir,
        collection_name=os.getenv("CHROMA_COLLECTION", "assignment_kb"),
        llm=llm,
    )

    def answer(user_message: str, chat_history, mem_state):
        user_message = (user_message or "").strip()
        if not user_message:
            return chat_history, mem_state

        chat_history = chat_history or []
        mem_state = mem_state or _memory_init_state(memory)
        low = user_message.lower()

        # Wikipedia forced
        if low.startswith(("wiki:", "wikipedia:")):
            topic = user_message.split(":", 1)[1].strip()
            reply = "[SOURCE: WIKIPEDIA]\n\n" + (
                wiki_summary_service(topic) if topic else "Try: `wiki: Mediterranean diet`"
            )
            chat_history.append([user_message, reply])
            mem_state = _update_memory(memory, mem_state, user_message, reply)
            return chat_history, mem_state

        # DOC forced => label TXT (ONLY when using doc:)
        if low.startswith(("doc:", "docs:", "txt:")):
            q = user_message.split(":", 1)[1].strip()
            if not q:
                reply = "Try: `doc: high protein breakfast`"
            else:
                txt = _semantic_answer(semantic, mem_state, q, k_default=4)
                reply = f"[SOURCE: TXT]\n\n{txt}"
            chat_history.append([user_message, reply])
            mem_state = _update_memory(memory, mem_state, user_message, reply)
            return chat_history, mem_state

        # Otherwise => semantic search but label RAG
        rag = _semantic_answer(semantic, mem_state, user_message, k_default=4) or ""
        rag_low = rag.strip().lower() if isinstance(rag, str) else ""

        rag_bad = (
            (not isinstance(rag, str)) or (not rag.strip())
            or ("doesn't contain" in rag_low)
            or ("does not contain" in rag_low)
            or ("not in the context" in rag_low)
            or ("no relevant" in rag_low)
            or ("no results" in rag_low)
            or ("⚠️ semanticservice error" in rag_low)
        )

        if not rag_bad:
            reply = f"[SOURCE: RAG]\n\n{rag}"
        else:
            reply = f"[SOURCE: LLM]\n\n{llm.chat(mem_state, user_message)}"

        chat_history.append([user_message, reply])
        mem_state = _update_memory(memory, mem_state, user_message, reply)
        return chat_history, mem_state

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n\n"
            "Use these prefixes to force a data source:\n"
            "- `wiki: Mediterranean diet`\n"
            "- `doc: high protein breakfast` (forces TXT/Chroma)\n\n"
            "Without prefixes: the bot tries RAG first, then falls back to LLM.\n"
        )
        chatbot = gr.Chatbot(height=450)
        msg = gr.Textbox(placeholder="Ask… (or use wiki: / doc:)")
        state = gr.State(_memory_init_state(memory))

        msg.submit(answer, [msg, chatbot, state], [chatbot, state])
        msg.submit(lambda: "", None, msg)

    return demo


if __name__ == "__main__":
    demo = build_app()

    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7860")),
        share=_env_bool("GRADIO_SHARE", default=False),
        show_error=True,
    )