from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import logging

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


    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""
        rag_prompt = build_rag_prompt(message, history, knowledge)

        print(f"rag prompt: {rag_prompt}")

        try:
            for response in llm.stream(rag_prompt):
                partial_message += response.content
                yield partial_message

        except Exception as e:
            logging.error(f"LLM error: {e}")
            yield "An error occurred while generating the response."

# initiate the Gradio app
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to the LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# launch the Gradio app
chatbot.launch()