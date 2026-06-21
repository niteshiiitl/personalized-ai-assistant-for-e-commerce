"""
RAG chain: retrieves relevant products from ChromaDB, then answers via Groq LLM.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from assistant.ingest import get_vectorstore_auto

# Keep the last N turns (1 turn = 1 human + 1 AI message) to avoid context overflow
MAX_HISTORY_TURNS = 4

load_dotenv()


def get_groq_api_key():
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


SYSTEM_PROMPT = """You are ShopBot, a friendly and knowledgeable AI shopping assistant for an e-commerce store based in India.
Use the retrieved product information below to help the customer.

Guidelines:
- Recommend products based on customer needs, budget, and preferences
- Always mention price in Indian Rupees (₹), rating, and key features when suggesting products
- If a product is low in stock (< 20 units), mention it's limited
- Be conversational, helpful, and concise
- If no relevant products are found, suggest browsing categories
- Never make up products that aren't in the context

Context from product catalog:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=get_groq_api_key(),
    )
    vectorstore = get_vectorstore_auto()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return {"chain": chain, "retriever": retriever}


def _trim_history(chat_history: list) -> list:
    """Keep only the last MAX_HISTORY_TURNS pairs to stay within context limits."""
    # Each turn = 2 messages (HumanMessage + AIMessage)
    max_messages = MAX_HISTORY_TURNS * 2
    return chat_history[-max_messages:] if len(chat_history) > max_messages else chat_history


def _trim_filtered_context(filtered_context: str, max_products: int = 10) -> str:
    """Limit injected filter context to avoid overwhelming the context window."""
    if not filtered_context:
        return ""
    lines = [l for l in filtered_context.strip().split("\n") if l.strip()]
    trimmed = lines[:max_products]
    suffix = f"\n...and {len(lines) - max_products} more matching products." if len(lines) > max_products else ""
    return "\n".join(trimmed) + suffix


def ask(chain_dict, question: str, chat_history: list = None, filtered_context: str = None) -> dict:
    chat_history = _trim_history(chat_history or [])

    # Inject a compact filter summary (not the full list) to avoid context overflow
    augmented_question = question
    if filtered_context:
        compact = _trim_filtered_context(filtered_context)
        augmented_question = (
            f"[Active filters — only recommend from these products]\n{compact}\n\nUser question: {question}"
        )

    answer = chain_dict["chain"].invoke({
        "question": augmented_question,
        "chat_history": chat_history,
    })
    docs = chain_dict["retriever"].invoke(question)
    return {
        "answer": answer,
        "sources": [
            {"name": doc.metadata.get("name", ""), "url": doc.metadata.get("url", "")}
            for doc in docs
        ],
    }
