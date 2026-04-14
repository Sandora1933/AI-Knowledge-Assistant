from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import logging

# import the .env file
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

# configuration
DATA_PATH = r"../data"
CHROMA_PATH = r"../chroma_db"
logging.info(f"DATA_PATH: {DATA_PATH}")
logging.info(f"CHROMA_PATH: {CHROMA_PATH}")

# initiate the embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
logging.info("Vector store initialized")

# loading the PDF document
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()
logging.info(f"Loaded {len(raw_documents)} raw documents")

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(raw_documents)
logging.info(f"Created {len(chunks)} chunks")

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]
logging.info(f"Generated {len(uuids)} UUIDs")

# adding chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)
logging.info(f"Documents added.")

