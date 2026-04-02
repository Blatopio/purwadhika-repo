from dotenv import load_dotenv
load_dotenv()
from import_data import embeddings, url_qdrant, qdrant_api_key
from langchain_qdrant import QdrantVectorStore

retriever = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="AmazonProducts",
    url=url_qdrant,
    api_key=qdrant_api_key,
)