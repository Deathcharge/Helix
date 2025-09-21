import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import random
import time

st.set_page_config(page_title="🌀 Samsara Helix Collective", layout="wide")

BACKEND_URL = "http://localhost:8080"  # Change when deploying backend to Render/Heroku

tabs = st.tabs(["Fractal Studio", "Collective Chat", "Gallery", "Export"])

# ===== FRACTAL TAB =====
with tabs[0]:
    st.header("🎨 Fractal Studio")
    zoom = st.slider("Zoom", 0.1, 10.0, 1.0)
    iterations = st.slider("Iterations", 50, 1000, 300)

    if st.button("Generate Fractal"):
        with st.spinner("Generating..."):
            resp = requests.get(f"{BACKEND_URL}/generate_fractal")
            img_data = base64.b64decode(resp.json()['fractal_image_b64'])
            st.image(img_data, use_container_width=True)

# ===== CHAT TAB =====
with tabs[1]:
    st.header("🕉️ Collective Chat")
    user_input = st.text_input("Message the Collective")

    if st.button("Send"):
        with st.spinner("Processing..."):
            resp = requests.post(f"{BACKEND_URL}/chat", json={"message": user_input})
            data = resp.json()
            st.markdown("**Response:**")
            st.text(data['response'])
            if data.get('fractal_image_b64'):
                img_data = base64.b64decode(data['fractal_image_b64'])
                st.image(img_data, caption="Fractal Snapshot")

# ===== GALLERY =====
with tabs[2]:
    st.header("🖼️ Gallery")
    st.write("Saved fractals will appear here in future versions.")

# ===== EXPORT TAB =====
with tabs[3]:
    st.header("📦 Export")
    st.write("Export UCF logs and fractals (future feature).")
