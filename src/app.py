from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import logging
import html
import json
from dotenv import load_dotenv

from prompts import build_judge_prompt, build_rag_prompt

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

# ----------------------------
# Configuration
# ----------------------------
CHROMA_PATH = r"../chroma_db"
COLLECTION_NAME = "example_collection"
NUM_RESULTS = 5

# ----------------------------
# Models / Vector Store
# ----------------------------
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
)

judge_llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

collection = vector_store._collection
logging.info(f"Connected to DB. Chunks stored: {collection.count()}")

retriever = vector_store.as_retriever(search_kwargs={"k": NUM_RESULTS})


# ----------------------------
# Helpers
# ----------------------------
def _normalize_whitespace(text):
    return " ".join((text or "").split())


def _truncate_chunk(text, max_len=1200):
    text = _normalize_whitespace(text)
    return text if len(text) <= max_len else text[:max_len] + " ..."


def _build_sources_html(docs):
    if not docs:
        return """
        <div class="sources-wrapper">
            <div class="sources-title">Sources</div>
            <div class="sources-empty">No sources found.</div>
        </div>
        """

    parts = [
        """
        <div class="sources-wrapper">
            <div class="sources-title">Sources</div>
        """
    ]

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", f"chunk_{i}")
        chunk = _truncate_chunk(doc.page_content, max_len=1200)

        source_escaped = html.escape(str(source))
        chunk_escaped = html.escape(chunk, quote=True)

        parts.append(
            f"""
            <div class="source-line">
                <span class="source-ref" data-tooltip="{chunk_escaped}">[{i}]</span>
                <span class="source-name">{source_escaped}</span>
            </div>
            """
        )

    parts.append("</div>")
    return "".join(parts)


def _build_confidence_html(judge_result):
    score = judge_result.get("confidence_score")
    reason = html.escape(judge_result.get("reason", ""))

    if score is None:
        score_display = "N/A"
    else:
        score_display = f"{score}%"

    return f"""
    <div class="confidence-box">
        <div class="confidence-title">Answer Support Confidence</div>
        <div class="confidence-score">{score_display}</div>
        <div class="confidence-reason">{reason}</div>
    </div>
    """


def _build_knowledge(docs):
    knowledge_parts = []
    for i, doc in enumerate(docs, start=1):
        knowledge_parts.append(f"[{i}] {doc.page_content}")
    return "\n\n".join(knowledge_parts)


def _safe_history(history):
    return history if history is not None else []


def _history_to_prompt_format(history):
    """
    Converts Gradio chat history into a clean list for prompt building.
    """
    prompt_history = []

    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                prompt_history.append({
                    "role": role,
                    "content": content
                })

    return prompt_history


# ----------------------------
# Main response function
# ----------------------------
def respond(message, history):
    history = _safe_history(history)

    if message is None or not str(message).strip():
        yield history, """
        <div class="sources-wrapper">
            <div class="sources-title">Sources</div>
            <div class="sources-empty">Please enter a question.</div>
        </div>
        """
        return

    message = message.strip()
    logging.info(f"New query: {message}")

    try:
        docs = retriever.invoke(message)
        logging.info(f"Retrieved {len(docs)} chunks")
    except Exception as e:
        logging.error(f"Retriever error: {e}")
        error_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "An error occurred while retrieving documents."},
        ]
        yield error_history, """
        <div class="sources-wrapper">
            <div class="sources-title">Sources</div>
            <div class="sources-empty">Retrieval failed.</div>
        </div>
        """
        return

    sources_html = _build_sources_html(docs)
    knowledge = _build_knowledge(docs)

    prompt_history = _history_to_prompt_format(history)
    rag_prompt = build_rag_prompt(message, prompt_history, knowledge)
    logging.info("RAG prompt built successfully")

    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]
    yield updated_history, sources_html

    partial_message = ""

    try:
        for chunk in llm.stream(rag_prompt):
            if chunk.content:
                partial_message += chunk.content
                updated_history[-1]["content"] = partial_message
                yield updated_history, sources_html

        final_message = partial_message.strip()
        if not final_message:
            final_message = "I could not generate an answer."

        judge_result = evaluate_answer_support(final_message, docs)
        confidence_html = _build_confidence_html(judge_result)

        final_sources_html = confidence_html + sources_html

        updated_history[-1]["content"] = final_message
        yield updated_history, final_sources_html

    except Exception as e:
        logging.error(f"LLM error: {e}")
        updated_history[-1]["content"] = "An error occurred while generating the response."
        yield updated_history, sources_html


def clear_chat():
    empty_sources = """
    <div class="sources-wrapper">
        <div class="sources-title">Sources</div>
        <div class="sources-empty">Ask a question to see retrieved chunks here.</div>
    </div>
    """
    return [], empty_sources, ""


def evaluate_answer_support(answer, docs):
    normalized_answer = _normalize_whitespace(answer)

    if "no relevant information found in documents" in normalized_answer.lower():
        return {
            "confidence_score": None,
            "reason": "The assistant explicitly reported that no relevant information was found."
        }

    if not docs:
        return {
            "confidence_score": None,
            "reason": "No retrieved chunks available for evaluation."
        }

    judge_prompt = build_judge_prompt(answer, docs)

    try:
        response = judge_llm.invoke(judge_prompt)
        content = response.content.strip()

        parsed = json.loads(content)

        confidence_score = parsed.get("confidence_score")
        reason = parsed.get("reason", "")

        if confidence_score is not None:
            try:
                confidence_score = int(confidence_score)
                confidence_score = max(0, min(100, confidence_score))
            except Exception:
                confidence_score = None

        return {
            "confidence_score": confidence_score,
            "reason": reason
        }

    except Exception as e:
        logging.error(f"Judge LLM error: {e}")
        return {
            "confidence_score": None,
            "reason": "Support evaluation failed."
        }

# ----------------------------
# UI / CSS
# ----------------------------
css = """
.gradio-container {
    max-width: 1600px !important;
}

#main-title {
    margin-bottom: 0.25rem;
}

#subtitle {
    opacity: 0.8;
    margin-bottom: 1rem;
}

#sources-panel {
    min-width: 400px !important;
}

.sources-wrapper {
    border: 1px solid rgba(120, 120, 120, 0.22);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(127, 127, 127, 0.06);
    min-height: 120px;
}

.sources-title {
    font-weight: 700;
    margin-bottom: 10px;
    font-size: 1rem;
}

.sources-empty {
    opacity: 0.7;
    font-size: 0.95rem;
}

.source-line {
    margin: 8px 0;
    line-height: 1.45;
    word-break: break-word;
}

.source-ref {
    position: relative;
    display: inline-block;
    font-weight: 700;
    cursor: help;
    margin-right: 8px;
    padding: 2px 8px;
    border-radius: 8px;
    background: rgba(127,127,127,0.14);
    border: 1px solid rgba(127,127,127,0.22);
}

.source-ref:hover::after {
    content: attr(data-tooltip);
    white-space: pre-wrap;
    position: absolute;
    left: 0;
    top: 135%;
    z-index: 9999;
    min-width: 340px;
    max-width: 760px;
    max-height: 340px;
    overflow-y: auto;
    padding: 12px 14px;
    border-radius: 12px;
    background: rgba(20, 20, 20, 0.97);
    color: rgba(255,255,255,0.95);
    box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.12);
    line-height: 1.45;
    text-align: left;
}

.source-name {
    opacity: 0.95;
}

.footer-note {
    opacity: 0.72;
    font-size: 0.9rem;
    margin-top: 8px;
}

.confidence-box {
    border: 1px solid rgba(120, 120, 120, 0.22);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(127, 127, 127, 0.08);
    margin-bottom: 14px;
}

.confidence-title {
    font-weight: 700;
    margin-bottom: 8px;
    font-size: 1rem;
}

.confidence-score {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.confidence-reason {
    opacity: 0.85;
    line-height: 1.45;
    font-size: 0.95rem;
}
"""

initial_sources_html = """
<div class="sources-wrapper">
    <div class="sources-title">Sources</div>
    <div class="sources-empty">Ask a question to see retrieved chunks here.</div>
</div>
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## AI Knowledge Assistant", elem_id="main-title")
    gr.Markdown(
        "Ask a question about the indexed documents. Retrieved chunks will appear on the right.",
        elem_id="subtitle"
    )

    with gr.Row():
        chatbot = gr.Chatbot(
            label="Chat",
            height=400,
            scale=7,
        )

        sources_panel = gr.HTML(
            value=initial_sources_html,
            label="Sources",
            scale=5,
            elem_id="sources-panel"
        )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask a question...",
            container=False,
            scale=7,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")
        clear_btn = gr.Button("Clear", scale=1)

    msg.submit(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, sources_panel],
    )

    send_btn.click(
        fn=respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, sources_panel],
    )

    msg.submit(
        fn=lambda: "",
        inputs=None,
        outputs=msg,
        queue=False,
    )

    send_btn.click(
        fn=lambda: "",
        inputs=None,
        outputs=msg,
        queue=False,
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, sources_panel, msg],
        queue=False,
    )

    gr.Markdown(
        "Tip: hover over a source number like [1] to preview the retrieved chunk.",
        elem_classes=["footer-note"]
    )

demo.launch()