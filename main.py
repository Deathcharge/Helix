import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import re
import time
import json
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db
import openai

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="🕉️ Samsara Helix v∞", layout="wide")

# -------------------------------
# Load Secrets
# -------------------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY
except KeyError:
    st.error("❌ OpenAI API key missing. Add OPENAI_API_KEY to Streamlit secrets.")

try:
    FIREBASE_KEY = json.loads(st.secrets["FIREBASE_KEY"])
    FIREBASE_DB_URL = "https://project-helix-f77a1-default-rtdb.firebaseio.com/"
except KeyError:
    st.error("❌ Firebase key missing. Add FIREBASE_KEY to Streamlit secrets.")
    FIREBASE_KEY = None

# -------------------------------
# Firebase Initialization
# -------------------------------
if FIREBASE_KEY and not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_KEY)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    except Exception as e:
        st.error(f"⚠️ Firebase initialization error: {e}")

# -------------------------------
# Initialize Session State
# -------------------------------
def init_session_state():
    defaults = {
        "chat_messages": [],
        "fractal_params": {
            "zoom": 1.0,
            "center_real": -0.7269,
            "center_imag": 0.1889,
            "iterations": 300,
            "width": 800,
            "height": 600,
            "colormap": "hot",
            "fractal_type": "mandelbrot",
            "color_invert": False
        },
        "audio_params": {
            "base_frequency": 432,
            "duration_sec": 5,
            "volume": 0.5,
            "waveform": "sine",
        },
        "animation_params": {
            "frame_count": 30,
            "zoom_step": 1.05,
        },
        "gallery_images": [],
        "settings": {
            "theme": "light",
            "language": "English",
            "auto_generate_fractal": False,
        },
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# -------------------------------
# Firebase Forum Functions
# -------------------------------
def create_thread(title, content, user):
    ref = db.reference("threads")
    new_thread = ref.push()
    new_thread.set({
        "title": title,
        "content": content,
        "user": user,
        "timestamp": time.time(),
        "replies": {}
    })

def add_reply(thread_id, reply_content, user):
    ref = db.reference(f"threads/{thread_id}/replies")
    new_reply = ref.push()
    new_reply.set({
        "content": reply_content,
        "user": user,
        "timestamp": time.time()
    })

def get_threads():
    ref = db.reference("threads")
    threads = ref.get()
    return threads or {}

# -------------------------------
# Fractal Generation
# -------------------------------
def generate_fractal(params):
    try:
        w, h = params["width"], params["height"]
        zoom = params["zoom"]
        cr, ci = params["center_real"], params["center_imag"]
        max_iter = params["iterations"]
        fractal_type = params["fractal_type"]
        cmap_name = params["colormap"]

        scale = 3.0 / zoom
        x = np.linspace(cr - scale, cr + scale, w)
        y = np.linspace(ci - scale, ci + scale, h)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        Z = np.zeros_like(C)
        output = np.zeros(C.shape, dtype=int)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            output[mask & (np.abs(Z) > 2)] = i

        norm = output / max_iter
        if params["color_invert"]:
            norm = 1 - norm
        cmap = plt.get_cmap(cmap_name)
        img = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        return img
    except Exception as e:
        st.error(f"Fractal generation error: {e}")
        return None

# -------------------------------
# Audio Generation
# -------------------------------
def generate_audio(params):
    fs = 44100
    t = np.linspace(0, params["duration_sec"], int(fs * params["duration_sec"]), False)
    freq = params["base_frequency"]

    if params["waveform"] == "sine":
        audio = np.sin(2 * np.pi * freq * t)
    elif params["waveform"] == "square":
        audio = np.sign(np.sin(2 * np.pi * freq * t))
    elif params["waveform"] == "sawtooth":
        audio = 2 * (t * freq - np.floor(0.5 + t * freq))
    else:
        audio = np.sin(2 * np.pi * freq * t)

    audio *= params["volume"]
    buffer = io.BytesIO()
    import soundfile as sf
    sf.write(buffer, audio, fs, format="WAV")
    buffer.seek(0)
    return buffer

# -------------------------------
# Helix Chat
# -------------------------------
def helix_response(user_input):
    try:
        messages = [
            {"role": "system", "content": "You are Helix, a multi-agent AI collective with rich contextual awareness."},
            {"role": "user", "content": user_input},
        ]
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, max_tokens=300)
        return resp.choices[0].message["content"]
    except Exception as e:
        return f"[Error generating response: {e}]"

# -------------------------------
# Forum UI
# -------------------------------
def forum_ui():
    st.header("Community Forum 🧵")
    username = st.text_input("Your Username")
    title = st.text_input("Thread Title")
    content = st.text_area("Thread Content")

    if st.button("Post Thread"):
        if username and title and content:
            create_thread(title, content, username)
            st.success("Thread posted!")
            st.experimental_rerun()
        else:
            st.error("All fields required.")

    st.subheader("Recent Threads")
    threads = get_threads()
    if threads:
        for thread_id, data in sorted(threads.items(), key=lambda x: x[1]["timestamp"], reverse=True):
            st.markdown(f"### {data['title']} (by {data['user']})")
            st.write(data["content"])
            st.caption(time.ctime(data["timestamp"]))

            if "replies" in data:
                for r_id, reply in data["replies"].items():
                    st.write(f"- **{reply['user']}**: {reply['content']}")

            reply_text = st.text_input(f"Reply to {thread_id}", key=f"reply-{thread_id}")
            if st.button(f"Send-{thread_id}"):
                if reply_text and username:
                    add_reply(thread_id, reply_text, username)
                    st.success("Reply added!")
                    st.experimental_rerun()

# -------------------------------
# Settings UI
# -------------------------------
def settings_ui():
    st.header("⚙️ Settings")
    theme = st.selectbox("Theme", ["light", "dark"], index=["light", "dark"].index(st.session_state.settings["theme"]))
    st.session_state.settings["theme"] = theme
    st.session_state.settings["language"] = st.selectbox("Language", ["English", "Sanskrit", "Other"])

# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs(["Fractal Studio", "Audio Synthesis", "Chat", "Gallery", "Forum", "Settings", "Export"])

# --- Fractal Studio ---
with tabs[0]:
    st.subheader("Fractal Studio")
    params = st.session_state.fractal_params
    params["zoom"] = st.slider("Zoom", 0.001, 500.0, params["zoom"])
    params["iterations"] = st.slider("Iterations", 10, 2000, params["iterations"])
    params["colormap"] = st.selectbox("Color Map", plt.colormaps(), index=plt.colormaps().index(params["colormap"]))
    if st.checkbox("Invert Colors", value=params["color_invert"]):
        params["color_invert"] = True
    if st.button("Generate Fractal"):
        img = generate_fractal(params)
        if img is not None:
            st.image(img, caption="Generated Fractal", use_container_width=True)
            st.session_state.gallery_images.append(img)

# --- Audio Synthesis ---
with tabs[1]:
    st.subheader("Audio Synthesis")
    params = st.session_state.audio_params
    params["base_frequency"] = st.slider("Base Frequency (Hz)", 20, 2000, params["base_frequency"])
    params["duration_sec"] = st.slider("Duration (sec)", 1, 20, params["duration_sec"])
    params["volume"] = st.slider("Volume", 0.0, 1.0, params["volume"])
    if st.button("Generate Audio"):
        audio_buffer = generate_audio(params)
        st.audio(audio_buffer, format="audio/wav")

# --- Chat ---
with tabs[2]:
    st.header("🕉️ Samsara Helix Chat")
    user_input = st.text_input("Type your message")
    if st.button("Send"):
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            response = helix_response(user_input)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Helix:** {msg['content']}")

# --- Gallery ---
with tabs[3]:
    st.subheader("Gallery")
    for i, img in enumerate(st.session_state.gallery_images):
        st.image(img, caption=f"Fractal {i+1}", use_container_width=True)

# --- Forum ---
with tabs[4]:
    forum_ui()

# --- Settings ---
with tabs[5]:
    settings_ui()

# --- Export ---
with tabs[6]:
    st.subheader("Export")
    if st.session_state.gallery_images:
        img = Image.fromarray(st.session_state.gallery_images[-1])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button("Download Last Fractal", data=buf.getvalue(), file_name="fractal.png", mime="image/png")
    else:
        st.info("No fractals to export.")
