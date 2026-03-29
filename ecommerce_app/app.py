"""
Streamlit E-Commerce App with Personalised AI Assistant (ShopBot)

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


# ── Helpers
# Reliable product images mapped by product name (Unsplash direct image URLs)
PRODUCT_IMAGES = {
    "Nike Air Max 270": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300&h=200&fit=crop",
    "Levi's 501 Jeans": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=300&h=200&fit=crop",
    "Apple AirPods Pro": "https://images.unsplash.com/photo-1600294037681-c80b4cb5b434?w=300&h=200&fit=crop",
    "Samsung 4K Smart TV 55inch": "https://images.unsplash.com/photo-1593359677879-a4bb92f829d1?w=300&h=200&fit=crop",
    "Instant Pot Duo 7-in-1": "https://images.unsplash.com/photo-1585515320310-259814833e62?w=300&h=200&fit=crop",
    "The Alchemist Book": "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=300&h=200&fit=crop",
    "Yoga Mat Premium": "https://images.unsplash.com/photo-1601925228008-f5e4c5e5b8e4?w=300&h=200&fit=crop",
    "Vitamin C Serum": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=300&h=200&fit=crop",
    "Mechanical Keyboard RGB": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=300&h=200&fit=crop",
    "Coffee Maker Drip": "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=300&h=200&fit=crop",
    "Running Shorts Nike": "https://images.unsplash.com/photo-1506629082955-511b1aa562c8?w=300&h=200&fit=crop",
    "Wireless Mouse Logitech": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=300&h=200&fit=crop",
    "Protein Powder Whey": "https://images.unsplash.com/photo-1593095948071-474c5cc2989d?w=300&h=200&fit=crop",
    "Sunglasses Polarized": "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=300&h=200&fit=crop",
    "Harry Potter Box Set": "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=300&h=200&fit=crop",
    "Sony WH-1000XM5": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=300&h=200&fit=crop",
    "iPad Air 11inch": "https://images.unsplash.com/photo-1544244015-0df4b3ffc6b0?w=300&h=200&fit=crop",
    "Adidas Ultraboost 22": "https://images.unsplash.com/photo-1608231387042-66d1773070a5?w=300&h=200&fit=crop",
    "Levi's Denim Jacket": "https://images.unsplash.com/photo-1551537482-f2075a1d41f2?w=300&h=200&fit=crop",
    "Air Fryer Ninja 6qt": "https://images.unsplash.com/photo-1585515320310-259814833e62?w=300&h=200&fit=crop",
    "Atomic Habits Book": "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=300&h=200&fit=crop",
    "Resistance Bands Set": "https://images.unsplash.com/photo-1598289431512-b97b0917affc?w=300&h=200&fit=crop",
    "Retinol Night Cream": "https://images.unsplash.com/photo-1556228578-8c89e6adf883?w=300&h=200&fit=crop",
    "Omega-3 Fish Oil": "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?w=300&h=200&fit=crop",
    "Leather Wallet Slim": "https://images.unsplash.com/photo-1627123424574-724758594e93?w=300&h=200&fit=crop",
    "Dell 27inch Monitor": "https://images.unsplash.com/photo-1527443224154-c4a3942d3acf?w=300&h=200&fit=crop",
    "Converse Chuck Taylor": "https://images.unsplash.com/photo-1463100099107-aa0980c362e6?w=300&h=200&fit=crop",
    "Hoodie Champion Reverse Weave": "https://images.unsplash.com/photo-1556821840-3a63f15732ce?w=300&h=200&fit=crop",
    "Blender Vitamix A2500": "https://images.unsplash.com/photo-1570197788417-0e82375c9371?w=300&h=200&fit=crop",
    "Rich Dad Poor Dad": "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=300&h=200&fit=crop",
    "Dumbbell Set Adjustable": "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=300&h=200&fit=crop",
    "Face Wash CeraVe": "https://images.unsplash.com/photo-1556228578-8c89e6adf883?w=300&h=200&fit=crop",
    "Creatine Monohydrate": "https://images.unsplash.com/photo-1593095948071-474c5cc2989d?w=300&h=200&fit=crop",
    "Backpack Osprey 40L": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=200&fit=crop",
    "Kindle Paperwhite": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=300&h=200&fit=crop",
    "New Balance 990v6": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300&h=200&fit=crop",
    "Linen Shirt Uniqlo": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=300&h=200&fit=crop",
    "Stand Mixer KitchenAid": "https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=300&h=200&fit=crop",
    "Atomic Habits Workbook": "https://images.unsplash.com/photo-1512820790803-83ca734da794?w=300&h=200&fit=crop",
    "Jump Rope Speed": "https://images.unsplash.com/photo-1598289431512-b97b0917affc?w=300&h=200&fit=crop",
    "Sunscreen SPF 50 EltaMD": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?w=300&h=200&fit=crop",
    "Multivitamin Men's": "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?w=300&h=200&fit=crop",
    "Luggage Samsonite 28inch": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=200&fit=crop",
    "Webcam Logitech 4K": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=300&h=200&fit=crop",
    "Hiking Boots Merrell": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300&h=200&fit=crop",
}

CATEGORY_FALLBACKS = {
    "Electronics": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=300&h=200&fit=crop",
    "Shoes": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=300&h=200&fit=crop",
    "Clothing": "https://images.unsplash.com/photo-1523381210434-271e8be1f52b?w=300&h=200&fit=crop",
    "Kitchen": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=300&h=200&fit=crop",
    "Books": "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=300&h=200&fit=crop",
    "Sports": "https://images.unsplash.com/photo-1517649763962-0c623066013b?w=300&h=200&fit=crop",
    "Beauty": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=300&h=200&fit=crop",
    "Health": "https://images.unsplash.com/photo-1584308666744-24d5c474f2ae?w=300&h=200&fit=crop",
    "Accessories": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=300&h=200&fit=crop",
}

def get_image_url(name: str, category: str) -> str:
    return PRODUCT_IMAGES.get(name) or CATEGORY_FALLBACKS.get(category) or "https://images.unsplash.com/photo-1472851294608-062f824d29cc?w=300&h=200&fit=crop"

def star_display(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty

@st.cache_data
def load_products():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")
    return pd.read_csv(csv_path)


#  Session state
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

#  Sidebar filter
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

#  Apply filter
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

#  Layers
chat_col, catalog_col = st.columns([3, 2], gap="large")

#  Chat panel 
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
                    link_html = " &nbsp;|&nbsp; ".join([
                        f'<a href="{s["url"]}" target="_blank" style="color:#e63946;text-decoration:none;font-weight:600;">{s["name"]}</a>'
                        for s in msg["sources"] if s.get("url")
                    ])
                    if link_html:
                        st.markdown(f'<div class="chat-sources">View products: {link_html}</div>', unsafe_allow_html=True)

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

#  Product catalog 
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
                            <a href="{product['url']}" target="_blank" style="display:inline-block;margin-top:8px;padding:5px 12px;background:#e63946;color:#fff;border-radius:6px;font-size:0.78rem;text-decoration:none;font-weight:600;">View on Amazon</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
