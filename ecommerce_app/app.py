"""
Streamlit E-Commerce App with Personalised AI Assistant (ShopBot)
Run: streamlit run ecommerce_app/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from assistant.rag_chain import build_chain, ask

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopSmart AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .product-card {
        background: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
    }
    .chat-bubble-user {
        background: #DCF8C6;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        text-align: right;
    }
    .chat-bubble-bot {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 10px 14px;
        margin: 6px 0;
        border: 1px solid #e0e0e0;
    }
    .low-stock { color: #e74c3c; font-weight: bold; }
    .rating-stars { color: #f39c12; }
</style>
""", unsafe_allow_html=True)

# ── Load product catalog ──────────────────────────────────────────────────────
@st.cache_data
def load_products():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")
    return pd.read_csv(csv_path)

# ── Init session state ────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    with st.spinner("Loading ShopBot... (first load may take a minute)"):
        try:
            st.session_state.chain = build_chain()
            st.session_state.chain_error = None
        except Exception as e:
            st.session_state.chain = None
            st.session_state.chain_error = str(e)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.get("chain_error"):
    st.error(f"Failed to load ShopBot: {st.session_state.chain_error}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛍️ ShopSmart AI")
    st.markdown("Your personal AI shopping assistant powered by RAG + Groq LLM.")
    st.divider()

    df = load_products()
    categories = ["All"] + sorted(df["category"].unique().tolist())
    selected_cat = st.selectbox("Browse by Category", categories)

    st.divider()
    st.markdown("**Quick Prompts**")
    quick_prompts = [
        "What electronics do you have under ₹10,000?",
        "Recommend a gift under ₹5,000",
        "What's your highest rated product?",
        "Show me sports & fitness items",
    ]
    for prompt in quick_prompts:
        if st.button(prompt, use_container_width=True):
            st.session_state.quick_input = prompt

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = build_chain()
        st.rerun()

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

# ── Chat panel ────────────────────────────────────────────────────────────────
with col1:
    st.subheader("💬 Chat with ShopBot")

    chat_container = st.container(height=480)
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Hi! I'm ShopBot. Ask me anything about our products — I'll help you find the perfect item!")
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                if msg.get("sources"):
                    st.caption(f"📦 Sources: {', '.join(msg['sources'])}")

    # Input
    default_input = st.session_state.pop("quick_input", "")
    user_input = st.chat_input("Ask ShopBot anything...", key="chat_input")

    # Handle quick prompt button clicks
    if default_input and not user_input:
        user_input = default_input

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # build chat history for context
        history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        with st.spinner("ShopBot is thinking..."):
            try:
                response = ask(st.session_state.chain, user_input, history)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, something went wrong: {e}",
                    "sources": [],
                })
        st.rerun()

# ── Product catalog panel ─────────────────────────────────────────────────────
with col2:
    st.subheader("🏪 Product Catalog")

    filtered_df = df if selected_cat == "All" else df[df["category"] == selected_cat]

    for _, row in filtered_df.iterrows():
        stock_label = f'<span class="low-stock">⚠️ Only {row["stock"]} left!</span>' if row["stock"] < 20 else f'✅ In Stock ({row["stock"]})'
        stars = "⭐" * int(round(row["rating"]))
        st.markdown(f"""
        <div class="product-card">
            <strong>{row['name']}</strong><br>
            <small>{row['category']}</small><br>
            <span class="rating-stars">{stars}</span> {row['rating']}/5<br>
            ₹<strong>{row['price']:,}</strong><br>
            {stock_label}<br>
            <small>{row['description']}</small>
        </div>
        """, unsafe_allow_html=True)
