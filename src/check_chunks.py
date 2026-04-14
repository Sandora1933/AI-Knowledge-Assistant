from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = r"../chroma_db"
COLLECTION_NAME = "example_collection"
TOP_K = 3

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

query = "How many encoders and decoders are used in the original Transformers architecture example?"

collection = vector_store._collection
results = vector_store.similarity_search(query, k=TOP_K)

print(f"Collection: {COLLECTION_NAME}")
print(f"Total chunks in DB: {collection.count()}")
print(f"\nQuery: {query}")

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Page: {doc.metadata.get('page')}")
    print("Content preview:")
    print(doc.page_content[:500])