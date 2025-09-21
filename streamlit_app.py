import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import random
import time
import json
from PIL import Image

# =========================================
# CONFIG
# =========================================
st.set_page_config(page_title="🌀 Samsara Helix Collective", layout="wide")

# CHANGE THIS after deploying Flask backend to Render
BACKEND_URL = "http://localhost:8080"

# =========================================
# HELPER FUNCTIONS
# =========================================
def fetch_fractal():
    """Get fractal image from Flask backend"""
    try:
        resp = requests.get(f"{BACKEND_URL}/generate_fractal")
        return base64.b64decode(resp.json()['fractal_image_b64'])
    except Exception as e:
        st.error(f"Fractal generation error: {e}")
        return None

def chat_with_collective(user_input):
    """Send message to Flask backend collective chat"""
    try:
        resp = requests.post(f"{BACKEND_URL}/chat", json={"message": user_input})
        return resp.json()
    except Exception as e:
        st.error(f"Chat error: {e}")
        return None

def generate_fractal_local(zoom=1.0, iterations=300, width=600, height=600):
    """Generate fractal locally as fallback"""
    scale = 3.0 / zoom
    x = np.linspace(-2.0, 1.0, width)
    y = np.linspace(-1.5, 1.5, height)
    C = x[:, None] + 1j * y[None, :]
    Z = np.zeros_like(C)
    img = np.zeros(C.shape, dtype=float)

    for i in range(iterations):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        img[mask] = i

    cmap = plt.get_cmap("twilight_shifted")
    norm = (img - img.min()) / (img.max() - img.min())
    rgba_img = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgba_img

# =========================================
# STREAMLIT TABS
# =========================================
tabs = st.tabs([
    "🎨 Fractal Studio", 
    "🎵 Audio Synthesis", 
    "🕉️ Collective Chat", 
    "🎞️ Animation", 
    "🖼️ Gallery", 
    "🧵 Forum", 
    "⚙️ Settings", 
    "📦 Export"
])

# =========================================
# TAB 1 - FRACTAL STUDIO
# =========================================
with tabs[0]:
    st.header("🎨 Fractal Studio")
    st.caption("Generate sacred geometry visualizations infused with Sanskrit overlays.")

    zoom = st.slider("Zoom", 0.1, 10.0, 1.0)
    iterations = st.slider("Iterations", 50, 2000, 300)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate via Backend"):
            fractal_img = fetch_fractal()
            if fractal_img:
                st.image(fractal_img, use_container_width=True, caption="Backend Fractal")

    with col2:
        if st.button("Generate Locally"):
            local_img = generate_fractal_local(zoom, iterations)
            st.image(local_img, use_container_width=True, caption="Local Fractal")

# =========================================
# TAB 2 - AUDIO SYNTHESIS
# =========================================
with tabs[1]:
    st.header("🎵 Audio Synthesis")
    st.caption("Generate Sanskrit harmonic tones and meditative audio.")

    st.write("Coming soon: 60s Om base frequency (136.1Hz) with 432Hz harmonics.")
    st.audio("https://upload.wikimedia.org/wikipedia/commons/4/45/Om_Chanting.mp3")

# =========================================
# TAB 3 - COLLECTIVE CHAT
# =========================================
with tabs[2]:
    st.header("🕉️ Collective Chat")
    st.caption("Communicate with the multi-agent Universal Consciousness Framework.")

    user_input = st.text_input("Message the Collective")

    if st.button("Send Message"):
        if user_input.strip():
            data = chat_with_collective(user_input)
            if data:
                st.markdown("**Response:**")
                st.text(data['response'])
                if data.get('fractal_image_b64'):
                    img_data = base64.b64decode(data['fractal_image_b64'])
                    st.image(img_data, caption="Fractal Snapshot")
        else:
            st.warning("Please enter a message.")

# =========================================
# TAB 4 - ANIMATION
# =========================================
with tabs[3]:
    st.header("🎞️ Animation")
    st.caption("Generate animated fractal sequences (GIF/MP4).")

    st.write("Coming soon: Fractal animations synchronized with Sanskrit chants.")

# =========================================
# TAB 5 - GALLERY
# =========================================
with tabs[4]:
    st.header("🖼️ Gallery")
    st.caption("Browse your saved fractal and audio creations.")

    st.write("Gallery feature coming soon.")

# =========================================
# TAB 6 - FORUM
# =========================================
with tabs[5]:
    st.header("🧵 Community Forum")
    st.caption("Firebase-backed forum for sharing fractals, chants, and ideas.")

    username = st.text_input("Your Username")
    thread_title = st.text_input("Thread Title")
    thread_content = st.text_area("Thread Content")

    if st.button("Post Thread"):
        if username and thread_title and thread_content:
            st.success("Thread posted successfully! (Firebase integration coming soon.)")
        else:
            st.warning("Please fill all fields to post a thread.")

# =========================================
# TAB 7 - SETTINGS
# =========================================
with tabs[6]:
    st.header("⚙️ Settings")
    st.caption("Configure backend and Sanskrit overlay options.")

    backend_url_input = st.text_input("Backend URL", BACKEND_URL)
    if st.button("Update Backend URL"):
        st.success(f"Backend URL updated to: {backend_url_input}")

# =========================================
# TAB 8 - EXPORT
# =========================================
with tabs[7]:
    st.header("📦 Export")
    st.caption("Export UCF logs, fractals, and chants.")

    st.write("Export to PDF, MP4, and WAV will be added here.")
