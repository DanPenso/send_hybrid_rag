#Composer through a Cursor IDE guided the development of this script

"""
Phase 4a: Chunk statutory text, embed with Cohere, index in ChromaDB for hybrid RAG retrieval.

Requires: Phase 1 outputs in data/02_extracted_text/, COHERE_API_KEY, Phase 4b uses this index.
"""
import os
import time
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- CONFIGURATION ---
INPUT_DIR = "../data/02_extracted_text/"
CHROMA_PATH = "../data/04_chroma_db_cohere"
COHERE_MODEL = "embed-english-v3.0"

#class to embed the documents with Cohere
class CohereEmbeddingFunctionFixed(EmbeddingFunction[Documents]):
    """Cohere v5 returns EmbedResponse; Chroma's bundled wrapper iterates it incorrectly."""
    #initializes the class
    def __init__(self, api_key: str, model_name: str = COHERE_MODEL) -> None:
        import cohere
        #initializes the client
        self._client = cohere.Client(api_key)
        self._model_name = model_name
    #embeds the documents
    def __call__(self, input: Documents) -> Embeddings:
        resp = self._client.embed(
            texts=list(input),
            model=self._model_name,
            input_type="search_document",
        )
        emb = resp.embeddings
        if isinstance(emb, list):
            return emb
        if emb is not None and getattr(emb, "float_", None) is not None:
            return emb.float_
        raise ValueError(f"Unexpected Cohere embed response: {type(resp)}")

#sets the cohere api key for the script
cohere_api_key = os.environ.get("COHERE_API_KEY")
if not cohere_api_key:
    raise ValueError("COHERE_API_KEY is required in .env")
#sets the cohere embedding function for the script
cohere_ef = CohereEmbeddingFunctionFixed(cohere_api_key, COHERE_MODEL)
#sets the chroma client for the script
client = chromadb.PersistentClient(path=CHROMA_PATH)
#sets the chroma collection for the script
collection = client.get_or_create_collection(
    name="send_statutory_cohere",
    embedding_function=cohere_ef,
)

#main function to run the script
def ingest_documents():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]

    print("Starting Cohere ingestion (Phase 4a)...")
#gets the existing documents from the chroma collection
    existing_docs = collection.get(include=["metadatas"])
    processed_files = set()
    #sets the processed files for the script
    if existing_docs["metadatas"]:
        #gets the metadata for the existing documents
        for meta in existing_docs["metadatas"]:
            processed_files.add(meta["source"])
    #iterates through the files
    for filename in all_files:
        if filename in processed_files:
            print(f"   Skip (indexed): {filename}")
            continue

        with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()

        chunks = text_splitter.split_text(text)
#sets the batch size for the script - 40 used because the Cohere API has a limit of 40 documents per request
        batch_size = 40
        for i in range(0, len(chunks), batch_size):
            chunk_batch = chunks[i : i + batch_size]
            id_batch = [f"{filename}_{i + j}" for j in range(len(chunk_batch))]
            meta_batch = [{"source": filename} for _ in range(len(chunk_batch))]
            #adds the documents to the chroma collection
            try:
                collection.add(documents=chunk_batch, metadatas=meta_batch, ids=id_batch)
                time.sleep(3)
            #handles the rate limit error
            except Exception:
                #prints the rate limit message
                print("   Rate limit: sleeping 60s...")
                time.sleep(60)
                collection.add(documents=chunk_batch, metadatas=meta_batch, ids=id_batch)
        #prints the number of chunks from the file
        print(f"   OK: {len(chunks)} chunks from {filename}")
        time.sleep(5)
    #prints the chroma db path
    print(f"\nChromaDB at {CHROMA_PATH}")

#main function to run the script
if __name__ == "__main__":
    ingest_documents()
