 Dietitian Chat Bot (TXT/RAG + Wikipedia) — Gradio + ChromaDB + OpenAI

IMPORTANT NOTE : I added these information addtional in 

.secrets file - template (new template with addiotnal information)
 API_GATEWAY_KEY=value
OPENAI_API_KEY=value
TAVILY_API_KEY=<Optional: Tavily API Key>
SQL_URL=postgresql://postgres:humanafterall@localhost:5432/reviews_db
NGROK_AUTHTOKEN=<Optional: NGROK Token>
LANGSMITH_API_KEY=<Optional Langsmit Key>
OPENAI_BASE_URL=<base_url>
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MODEL=gpt-4o-mini
API_GATEWAY_HEADER=x-api-key

A simple chatbot app that can answer diet/nutrition questions using:

- **Wikipedia API** (forced with `wiki:` prefix)
- **Local TXT knowledge base via RAG** (forced with `doc:` prefix)
- **LLM fallback** (when RAG cannot answer)

Built with Gradio UI, ChromaDB for vector search, and an OpenAI model for responses.

---

 1) USE CASE

 User can ask in 3 ways

1) **Wikipedia mode (forced)**
   - User: `wiki: Mediterranean diet`
   - Bot: calls `wiki_summary_service(topic)` and returns a summary.

2) **TXT mode (forced document/RAG)**
   - User: `doc: high protein breakfast ideas`
   - Bot: queries ChromaDB (your embedded TXT knowledge base) and returns relevant info.

3) **Default mode (automatic)**
   - User: `What should I eat after workout?`
   - Bot behavior:
     - Try **RAG** first (`SemanticService`)
     - If RAG has no good result → fallback to **LLM** response

---

 2) Tech stack / Tools used

 Core app tools
- Python
- Gradio (Chat UI)
- FastAPI/Uvicorn (Gradio runs on ASGI)
- ChromaDB (vector database for RAG)
- OpenAI model (chat model for LLM + optionally embeddings depending on your SemanticService)
- Wikipedia API / Wikipedia wrapper (your `wiki_summary_service`)

 project services
- LLMChat → handles model calls with system prompt/persona
- SemanticService → handles retrieval from ChromaDB + (optional) re-ranking/answer formatting
- wiki_summary_service → fetches wiki content
- MemoryManager` → stores summary + conversation turns

---

 3) Folder structure 

Example structure based on your repo:

05_src/
assignment_chat/
app.py
memory.py
utils_secrets.py
services/
llm_chat.py
api_service.py # wiki_summary_service here
semantic_service.py # RAG / ChromaDB logic here
data/
chroma_store/ # Chroma persistent directory



### Important folders
- `data/chroma_store/`  
  This is where ChromaDB persists embeddings and collections.

---

 4) Architecture overview

 High-level flow

mermaid
flowchart TD
    U[User] -->|message| G[Gradio UI]
    G --> A[answer() router in app.py]

    A -->|prefix wiki:| W[wiki_summary_service]
    A -->|prefix doc:| R1[SemanticService (TXT Retrieval)]
    A -->|no prefix| R2[Try RAG first]

    R2 -->|good result| OUT1[Return SOURCE: RAG]
    R2 -->|no result| L[LLMChat (OpenAI)]
    L --> OUT2[Return SOURCE: LLM]

    W --> OUT3[Return SOURCE: WIKIPEDIA]
    R1 --> OUT4[Return SOURCE: TXT]

5) Detailed step-by-step: How the app works
Step 1 — Load environment & secrets

In build_app():

Your app loads .secrets using apply_secrets_to_env()

This sets variables like:

OPENAI_API_KEY

OPENAI_MODEL

optional: CHROMA_COLLECTION

Step 2 — Initialize LLM + Memory
llm = LLMChat(model=model, system_persona=SYSTEM_PERSONA)
memory = MemoryManager(llm=llm)

SYSTEM_PERSONA defines style and safety rules.

MemoryManager keeps conversation state.

Step 3 — Initialize SemanticService (RAG)
semantic = SemanticService(
    persist_dir=".../data/chroma_store",
    collection_name="assignment_kb",
    llm=llm,
)

persist_dir must point to your ChromaDB folder.

collection_name must match your actual collection
You confirmed your DB has:

assignment_kb => count: 20

So the app uses assignment_kb, not assignment_chat.

Step 4 — UI components (Gradio)
with gr.Blocks(...) as demo:
    chatbot = gr.Chatbot(...)
    msg = gr.Textbox(...)
    state = gr.State(...)

msg.submit(answer, ...) wires textbox input to your answer() function.

state stores memory state across turns.

Step 5 — Message routing logic (answer())

The router checks prefixes:

A) Wikipedia forced route

If message starts with wiki::

Extract topic

Call wiki_summary_service(topic)

Return [SOURCE: WIKIPEDIA]

B) TXT forced route

If message starts with doc::

Extract query

Call SemanticService.answer(...)

Return [SOURCE: TXT]

C) Default route

If no prefix:

Try semantic answer (RAG)

If RAG looks good → [SOURCE: RAG]

Else → LLM fallback [SOURCE: LLM]

6) RAG design (how retrieval work

What is RAG?

RAG = Retrieval-Augmented Generation.

Instead of letting the model guess, we:

Retrieve relevant chunks from a knowledge base (TXT docs embedded into ChromaDB)

Use those results to answer (either directly or using LLM guided by retrieved context)

RAG pipeline diagram








Key components in your app

ChromaDB collection: assignment_kb

Top-k retrieval: k_default=4 (you can change)

SemanticService.answer():

likely does:

embed query

similarity search

return best matching chunk(s)

“TXT” vs “RAG” label in your UI

When user explicitly types doc: → label is [SOURCE: TXT]

When user doesn’t specify and the system retrieves automatically → label is [SOURCE: RAG]

This makes testing easy.


) How to TEST (Proof that each source works)
Test A — Wikipedia

Ask:

wiki: Mediterranean diet

Expected:

Response starts with:
[SOURCE: WIKIPEDIA]

Test B — TXT/Chroma retrieval

Ask something that should exist in your assignment_kb docs, like:

doc: list foods high in fiber

doc: what is a balanced breakfast

doc: diet tips for diabetes (if those words exist in your KB)

Expected:

Response starts with:
[SOURCE: TXT]

And it should contain content clearly coming from your documents.

Test C — Default mode auto-RAG

Ask:

What are healthy snacks?

Expected:

If KB has relevant matches, you’ll get:
[SOURCE: RAG]

) How to TEST (Proof that each source works)
Test A — Wikipedia

Ask:

wiki: Mediterranean diet

Expected:

Response starts with:
[SOURCE: WIKIPEDIA]

Test B — TXT/Chroma retrieval

Ask something that should exist in your assignment_kb docs, like:

doc: list foods high in fiber

doc: what is a balanced breakfast

doc: diet tips for diabetes (if those words exist in your KB)

Expected:

Response starts with:
[SOURCE: TXT]

And it should contain content clearly coming from your documents.

Test C — Default mode auto-RAG

Ask:

What are healthy snacks?

Expected:

If KB has relevant matches, you’ll get:
[SOURCE: RAG]



8) Common issues and fixes
Issue 1: Chroma collection is empty / wrong name

Symptoms:

RAG returns nothing

col.count() is 0

Fix:

Confirm collection name:python -c "import chromadb; p=r'...\\data\\chroma_store'; c=chromadb.PersistentClient(path=p); print([col.name for col in c.list_collections()])"

Then set:

collection_name="assignment_kb" (or correct value)

Issue 2: “dict has no attribute strip”

Cause:

SemanticService expected question first, but got mem_state first.

Fix:

We updated _semantic_answer() to try multiple call signatures and coerce output.

Issue 3: gradio_client schema bool error (bool has no attribute get)

Cause:

Some JSON schemas can be True/False (valid JSON schema)

gradio_client wasn’t handling it in your version

Fix:

app.py patches gradio_client.utils before importing gradio.

Issue 4: localhost not accessible (share=True forced)

Cause:

Proxy settings interfere with localhost detection

Fix:

Set NO_PROXY to include localhost:
os.environ.setdefault("NO_PROXY","localhost,127.0.0.1,0.0.0.0")


9) How to run the app
  1) Activate env
  .\deploying-ai-env\Scripts\Activate.ps1
  2) RUN 
  python .\05_src\assignment_chat\app.py

  Open:
  http://127.0.0.1:7860

