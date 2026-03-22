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

load_dotenv()


def get_groq_api_key():
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
        model="llama3-8b-8192",
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


def ask(chain_dict, question: str, chat_history: list = None) -> dict:
    chat_history = chat_history or []
    answer = chain_dict["chain"].invoke({
        "question": question,
        "chat_history": chat_history,
    })
    # get source docs for citations
    docs = chain_dict["retriever"].invoke(question)
    return {
        "answer": answer,
        "sources": [doc.metadata.get("name", "") for doc in docs],
    }
