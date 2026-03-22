"""
RAG chain: retrieves relevant products from ChromaDB, then answers via Groq LLM.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from assistant.ingest import get_vectorstore_auto

load_dotenv()

def get_groq_api_key():
    # Try Streamlit secrets first (Streamlit Cloud), fall back to .env
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """You are ShopBot, a friendly and knowledgeable AI shopping assistant for an e-commerce store.
Use the retrieved product information below to help the customer.

Guidelines:
- Recommend products based on customer needs, budget, and preferences
- Always mention price, rating, and key features when suggesting products
- If a product is low in stock (< 20 units), mention it's limited
- Be conversational, helpful, and concise
- If no relevant products are found, suggest browsing categories
- Never make up products that aren't in the context

Context from product catalog:
{context}

Chat History:
{chat_history}

Customer Question: {question}

ShopBot Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_PROMPT,
)


def build_chain():
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.3,
        groq_api_key=get_groq_api_key(),
    )

    vectorstore = get_vectorstore_auto()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},  # top 4 relevant products
    )

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5,  # remember last 5 exchanges
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain


def ask(chain, question: str) -> dict:
    result = chain({"question": question})
    return {
        "answer": result["answer"],
        "sources": [doc.metadata["name"] for doc in result.get("source_documents", [])],
    }
