"""
Ingest product catalog CSV into ChromaDB vector store.
Run this once before starting the app: python -m assistant.ingest
"""

import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Paths relative to project root (works locally and on Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTS_CSV = os.path.join(BASE_DIR, "data", "products.csv")
CHROMA_DIR = os.path.join(BASE_DIR, "vectorstore")
COLLECTION_NAME = "products"


def get_vectorstore_auto():
    """Build vectorstore if it doesn't exist, then return it."""
    if not os.path.exists(CHROMA_DIR):
        build_vectorstore()
    return get_vectorstore()


def load_products(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = (
            f"Product: {row['name']}\n"
            f"Category: {row['category']}\n"
            f"Price: ${row['price']}\n"
            f"Description: {row['description']}\n"
            f"Stock: {row['stock']} units\n"
            f"Rating: {row['rating']}/5"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "id": str(row["id"]),
                "name": row["name"],
                "category": row["category"],
                "price": float(row["price"]),
                "rating": float(row["rating"]),
                "stock": int(row["stock"]),
            }
        ))
    return docs


def build_vectorstore():
    print("Loading products...")
    docs = load_products(PRODUCTS_CSV)

    print("Building embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Done. {len(docs)} products indexed into ChromaDB at '{CHROMA_DIR}'")
    return vectorstore


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


if __name__ == "__main__":
    build_vectorstore()
