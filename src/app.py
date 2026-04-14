from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import logging
import html

# import the .env file
from dotenv import load_dotenv

from prompts import build_rag_prompt

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

# configuration
DATA_PATH = r"../data"
CHROMA_PATH = r"../chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the model
llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# connect to the chromadb
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

collection = vector_store._collection
logging.info(f"Connected to DB. Chunks stored: {collection.count()}")

# Set up the vectorstore to be the retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# --- citations / tooltips ---
def _normalize_whitespace(s: str) -> str:
    return " ".join((s or "").split())


def _truncate(s: str, max_len: int) -> str:
    s = s or ""
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _build_citations_markdown(docs) -> str:
    if not docs:
        return ""

    lines = ["**Sources:**"]
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", f"chunk_{i}")
        chunk = _truncate(_normalize_whitespace(doc.page_content), 180)
        lines.append(f"- [{i}] **{source}**: {chunk}")

    return "\n".join(lines)


# call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input: {message}. History: {history}\n")
    logging.info(f"New query: {message}")

    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)
    logging.info(f"Retrieved {len(docs)} chunks")

    # add all the chunks to 'knowledge'
    knowledge = ""

    sources = []

    for i, doc in enumerate(docs):
        knowledge += f"[{i}] {doc.page_content}\n\n"
        sources.append(doc.metadata.get("source", f"chunk_{i}"))

    citations_html = _build_citations_markdown(docs)

    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""
        rag_prompt = build_rag_prompt(message, history, knowledge)

        print(f"rag prompt: {rag_prompt}")

        try:
            for response in llm.stream(rag_prompt):
                partial_message += response.content
                yield partial_message

            # final message: append citations after streaming finishes
            final_message = partial_message.rstrip()
            if citations_html:
                final_message += "\n\n" + citations_html
            yield final_message

        except Exception as e:
            logging.error(f"LLM error: {e}")
            yield "An error occurred while generating the response."

# initiate the Gradio app
css = """
.citations { margin-top: 0.65rem; display: flex; gap: 0.35rem; flex-wrap: wrap; align-items: center; }
.citations-label { opacity: 0.8; font-size: 0.9em; margin-right: 0.25rem; }
.citation {
  position: relative;
  display: inline-flex;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 0.92em;
  line-height: 1;
  padding: 0.15rem 0.35rem;
  border: 1px solid rgba(127,127,127,0.35);
  border-radius: 0.5rem;
  background: rgba(127,127,127,0.10);
  cursor: help;
  user-select: none;
}
.citation::after {
  content: attr(data-tooltip);
  white-space: pre-wrap;
  display: none;
  position: absolute;
  z-index: 1000;
  left: 0;
  top: 140%;
  min-width: 280px;
  max-width: 560px;
  padding: 0.65rem 0.75rem;
  border-radius: 0.75rem;
  border: 1px solid rgba(127,127,127,0.35);
  background: rgba(22,22,22,0.98);
  color: rgba(255,255,255,0.92);
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.citation:hover::after { display: block; }
"""

with gr.Blocks(css=css) as demo:
    chatbot = gr.ChatInterface(
        stream_response,
        chatbot=gr.Chatbot(sanitize_html=False),
        textbox=gr.Textbox(
            placeholder="Send to the LLM...",
            container=False,
            scale=7,
        ),
    )

demo.launch()