import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_openai import OpenAIEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from langchain_qdrant import QdrantVectorStore

from langchain_core.documents import Document

from uuid import uuid4

data_path = r"C:\Alghi\Boothcamp\Purwadhika\Practice Session\Modul-3_session-11\purwadhika-repo\shopping-assistant\amazon.csv"

# read data from .csv file
df = pd.read_csv(data_path)
df= df.head(100)
# show data
print(df)

# define embedding engine for vector database

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# creating document for each product description

documents = []

for i in range(len(df)):
  product_name = df["product_name"][i]
  product_description = df["description"][i]
  product_id = df["uniq_id"][i]
  doc = Document(
      page_content=product_description,
      metadata={
          "product_id" : product_id,
          "product_name" : product_name,
      },
  )
  documents.append(doc)

# set up unique id for each document
uuids = [str(uuid4()) for _ in range(len(documents))]

qdrant_api_key = os.getenv("QDRANT_API_KEY")
url_qdrant = os.getenv("QDRANT_URL")

qdrant = QdrantVectorStore.from_documents(
    documents,
    embedding=embeddings,
    url=url_qdrant,
    prefer_grpc=True,
    api_key=qdrant_api_key,
    collection_name="AmazonProducts",
)

client = QdrantClient(
    url=url_qdrant,
    api_key=qdrant_api_key
)

response = client.get_collections()
print(response)

