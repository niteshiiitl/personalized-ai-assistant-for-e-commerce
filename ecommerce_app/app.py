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

st.set_page_config(
    page_title="ShopSmart AI",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global */
    body { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; }

    /* Product card */
    .product-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 0;
        margin-bottom: 16px;
        border: 1px solid #e8e8e8;
        overflow: hidden;
        transition: box-shadow 0.2s;
    }
    .product-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
    .product-info { padding: 12px 14px 14px 14px; }
    .product-name { font-size: 0.95rem; font-weight: 600; color: #1a1a1a; margin-bottom: 2px; }
    .product-category { font-size: 0.75rem; color: #888; margin-bottom: 6px; }
    .product-price { font-size: 1.1rem; font-weight: 700; color: #e63946; }
    .product-rating { font-size: 0.8rem; color: #f4a261; margin-top: 4px; }
    .low-stock { font-size: 0.75rem; color: #e63946; font-weight: 600; margin-top: 4px; }
    .in-stock { font-size: 0.75rem; color: #2a9d8f; margin-top: 4px; }

    /* Chat */
    .chat-user {
        background: #f0f4ff;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 14px;
        margin: 6px 0 6px 40px;
        font-size: 0.9rem;
        color: #1a1a1a;
    }
    .chat-bot {
        background: #f9f9f9;
        border-radius: 12px 12px 12px 2px;
        padding: 10px 14px;
        margin: 6px 40px 6px 0;
        font-size: 0.9rem;
        border: 1px solid #e8e8e8;
        color: #1a1a1a;
    }
    .chat-sources { font-size: 0.75rem; color: #888; margin-top: 4px; }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 2px solid #e63946;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_image_url(name: str, category: str) -> str:
    query = name.replace(" ", "+")
    return f"https://source.unsplash.com/300x200/?{query}"

def star_display(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty

@st.cache_data
def load_products():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")
    return pd.read_csv(csv_path)


# ── Session state ─────────────────────────────────────────────────────────────
if "chain" not in st.session_state:
    with st.spinner("Loading ShopBot..."):
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

df = load_products()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ShopSmart AI")
    st.caption("Powered by RAG + Groq LLaMA 3.3")
    st.divider()

    st.markdown("**Filters**")
    categories = ["All"] + sorted(df["category"].unique().tolist())
    selected_cat = st.selectbox("Category", categories)

    price_min, price_max = int(df["price"].min()), int(df["price"].max())
    price_range = st.slider("Price Range (₹)", price_min, price_max, (price_min, price_max), step=500)

    min_rating = st.slider("Minimum Rating", 1.0, 5.0, 4.0, step=0.1)

    st.divider()
    st.markdown("**Quick Prompts**")
    quick_prompts = [
        "What electronics are under ₹10,000?",
        "Recommend a gift under ₹5,000",
        "What is your highest rated product?",
        "Show me sports and fitness items",
    ]
    for prompt in quick_prompts:
        if st.button(prompt, use_container_width=True):
            st.session_state.quick_input = prompt

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain = build_chain()
        st.rerun()

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered_df = df.copy()
if selected_cat != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_cat]
filtered_df = filtered_df[
    (filtered_df["price"] >= price_range[0]) &
    (filtered_df["price"] <= price_range[1]) &
    (filtered_df["rating"] >= min_rating)
]

# Build filtered context for RAG
filtered_context = "\n\n".join([
    f"Product: {row['name']} | Category: {row['category']} | Price: ₹{row['price']} | Rating: {row['rating']}/5 | Stock: {row['stock']}"
    for _, row in filtered_df.iterrows()
])

# ── Layout ────────────────────────────────────────────────────────────────────
chat_col, catalog_col = st.columns([3, 2], gap="large")

# ── Chat panel ────────────────────────────────────────────────────────────────
with chat_col:
    st.markdown('<span class="section-title">Chat with ShopBot</span>', unsafe_allow_html=True)

    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.messages:
            st.info("Hi! I'm ShopBot. Ask me anything about our products and I'll help you find the perfect item.")
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)
                if msg.get("sources"):
                    st.markdown(f'<div class="chat-sources">Sources: {", ".join(msg["sources"])}</div>', unsafe_allow_html=True)

    default_input = st.session_state.pop("quick_input", "")
    user_input = st.chat_input("Ask ShopBot anything...")
    if default_input and not user_input:
        user_input = default_input

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        with st.spinner("ShopBot is thinking..."):
            try:
                response = ask(st.session_state.chain, user_input, history, filtered_context)
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

# ── Product catalog ───────────────────────────────────────────────────────────
with catalog_col:
    st.markdown(
        f'<span class="section-title">Product Catalog</span> <span style="font-size:0.8rem;color:#888;">({len(filtered_df)} items)</span>',
        unsafe_allow_html=True
    )

    if filtered_df.empty:
        st.warning("No products match your filters.")
    else:
        # Display 2 products per row
        rows = [filtered_df.iloc[i:i+2] for i in range(0, len(filtered_df), 2)]
        for row_df in rows:
            cols = st.columns(2)
            for col, (_, product) in zip(cols, row_df.iterrows()):
                with col:
                    img_url = get_image_url(product["name"], product["category"])
                    stock_html = (
                        f'<div class="low-stock">Only {product["stock"]} left</div>'
                        if product["stock"] < 20
                        else f'<div class="in-stock">In Stock ({product["stock"]})</div>'
                    )
                    st.markdown(f"""
                    <div class="product-card">
                        <img src="{img_url}" style="width:100%;height:140px;object-fit:cover;" onerror="this.src='https://via.placeholder.com/300x140?text=No+Image'"/>
                        <div class="product-info">
                            <div class="product-name">{product['name']}</div>
                            <div class="product-category">{product['category']}</div>
                            <div class="product-price">₹{product['price']:,}</div>
                            <div class="product-rating">{star_display(product['rating'])} {product['rating']}</div>
                            {stock_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
