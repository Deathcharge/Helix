import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import time
import json
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db

# ==============================
# CONFIGURATION
# ==============================
st.set_page_config(page_title="🕉️ Samsara Helix v∞ Limitless", layout="wide")

FIREBASE_DB_URL = "https://project-helix-f77a1-default-rtdb.firebaseio.com/"

# Load Firebase key from secrets
try:
    firebase_key_dict = json.loads(st.secrets["FIREBASE_KEY"])
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_key_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
except KeyError:
    st.error("❌ Firebase key not found in Streamlit secrets.")
except Exception as e:
    st.error(f"⚠️ Firebase initialization error: {e}")

# ==============================
# SESSION STATE INITIALIZATION
# ==============================
def init_session_state():
    defaults = {
        "username": "",
        "chat_messages": [],
        "chat_input": "",
        "rerun_flag": False,
        "fractal_params": {
            "zoom": 1.0,
            "center_real": -0.7269,
            "center_imag": 0.1889,
            "iterations": 100,
            "width": 600,
            "height": 450,
            "colormap": "hot"
        },
        "audio_params": {
            "base_frequency": 432,
            "duration_sec": 5,
            "volume": 0.5,
            "waveform": "sine"
        },
        "animation_params": {
            "frame_count": 30,
            "width": 400,
            "height": 400,
            "zoom": 1.0,
            "center_real": -0.7269,
            "center_imag": 0.1889,
            "iterations": 100,
            "colormap": "hot"
        },
        "gallery_images": [],
        "settings": {
            "theme": "light",
            "language": "English",
            "auto_generate_fractal": False
        }
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ==============================
# FIREBASE FUNCTIONS
# ==============================
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

# ==============================
# FRACTAL GENERATION
# ==============================
def generate_fractal(params):
    width, height = params["width"], params["height"]
    zoom = params["zoom"]
    center_real = params["center_real"]
    center_imag = params["center_imag"]
    max_iter = params["iterations"]
    colormap = params["colormap"]

    scale = 3.0 / zoom
    x_min = center_real - scale / 2
    x_max = center_real + scale / 2
    y_min = center_imag - scale / 2 * height / width
    y_max = center_imag + scale / 2 * height / width

    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    escape_time = np.zeros(C.shape, dtype=float)

    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask] ** 2 + C[mask]
        escaped = (np.abs(Z) > 2) & (escape_time == 0)
        escape_time[escaped] = i + 1 - np.log2(np.log2(np.abs(Z[escaped])))

    escape_time[escape_time == 0] = max_iter
    norm = (escape_time - escape_time.min()) / (escape_time.max() - escape_time.min())
    cmap = plt.get_cmap(colormap)
    colored = cmap(norm)
    return (colored[:, :, :3] * 255).astype(np.uint8)

# ==============================
# AUDIO SYNTHESIS
# ==============================
def generate_audio(params):
    import soundfile as sf
    fs = 44100
    t = np.linspace(0, params["duration_sec"], int(fs * params["duration_sec"]), False)
    freq = params["base_frequency"]
    waveform = params["waveform"]

    if waveform == "sine":
        audio = np.sin(2 * np.pi * freq * t)
    elif waveform == "square":
        audio = np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform == "sawtooth":
        audio = 2 * (t * freq - np.floor(0.5 + t * freq))
    else:
        audio = np.sin(2 * np.pi * freq * t)

    audio *= params["volume"]
    audio = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    sf.write(buffer, audio, fs, format="WAV")
    buffer.seek(0)
    return buffer

# ==============================
# ANIMATION GENERATION
# ==============================
def generate_animation(params):
    import imageio
    frames = []
    cmap = plt.get_cmap(params["colormap"])
    for i in range(params["frame_count"]):
        zoom = params["zoom"] * (1 + i / params["frame_count"])
        temp_params = params.copy()
        temp_params["zoom"] = zoom
        img_array = generate_fractal(temp_params)
        frames.append(img_array)
    buffer = io.BytesIO()
    imageio.mimsave(buffer, frames, format="GIF", duration=0.1)
    buffer.seek(0)
    return buffer

# ==============================
# CHAT UI
# ==============================
def chat_ui():
    st.header("🕉️ Samsara Helix Chat")

    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Helix:** {msg['content']}")

    def send_chat_message():
        user_msg = st.session_state.chat_input.strip()
        if not user_msg:
            return
        st.session_state.chat_messages.append({"role": "user", "content": user_msg})
        # Basic echo for now
        st.session_state.chat_messages.append({"role": "assistant", "content": f"You said: {user_msg}"})
        st.session_state.chat_input = ""

    st.text_input("Type your message here...", key="chat_input")
    st.button("Send", on_click=send_chat_message)

# ==============================
# FORUM UI
# ==============================
def forum_ui():
    st.header("Community Forum 🧵")
    st.session_state.username = st.text_input("Your Username", value=st.session_state.username)

    st.subheader("Create New Thread")
    title = st.text_input("Thread Title")
    content = st.text_area("Thread Content")

    if st.button("Post Thread"):
        if title and content and st.session_state.username:
            create_thread(title, content, st.session_state.username)
            st.success("Thread posted successfully!")
            st.rerun()
        else:
            st.error("Please provide title, content, and username.")

    st.subheader("🔥 Recent Threads")
    threads = get_threads()
    if not threads:
        st.info("No threads yet.")
    else:
        for thread_id, thread_data in sorted(threads.items(), key=lambda x: x[1]["timestamp"], reverse=True):
            st.markdown(f"### {thread_data['title']} (by {thread_data['user']})")
            st.write(thread_data['content'])
            st.caption(f"Posted: {time.ctime(thread_data['timestamp'])}")

            reply_content = st.text_input(f"Reply to thread {thread_id}", key=f"reply-{thread_id}")
            if st.button(f"Reply-{thread_id}"):
                if reply_content and st.session_state.username:
                    add_reply(thread_id, reply_content, st.session_state.username)
                    st.success("Reply added!")
                    st.rerun()

# ==============================
# SETTINGS UI
# ==============================
def settings_ui():
    st.header("⚙️ Settings")
    st.session_state.settings["theme"] = st.selectbox("Theme", ["light", "dark"], index=0)
    st.session_state.settings["language"] = st.selectbox("Language", ["English", "Sanskrit", "Other"])
    st.session_state.settings["auto_generate_fractal"] = st.checkbox(
        "Auto-generate fractals on parameter change", value=st.session_state.settings["auto_generate_fractal"]
    )

# ==============================
# TAB LAYOUT
# ==============================
tabs = st.tabs(["Fractal Studio", "Audio Synthesis", "Chat", "Animation", "Gallery", "Forum", "Settings", "Export"])

with tabs[0]:
    st.subheader("Fractal Studio")
    params = st.session_state.fractal_params
    params["zoom"] = st.slider("Zoom", 0.0001, 2000.0, params["zoom"])
    params["iterations"] = st.slider("Iterations", 10, 1000, params["iterations"])
    params["colormap"] = st.selectbox("Color Map", plt.colormaps(), index=plt.colormaps().index(params["colormap"]))
    if st.button("Generate Fractal"):
        st.image(generate_fractal(params), use_container_width=True)

with tabs[1]:
    st.subheader("Audio Synthesis")
    audio_buffer = generate_audio(st.session_state.audio_params)
    if st.button("Generate Audio"):
        st.audio(audio_buffer, format="audio/wav")

with tabs[2]:
    chat_ui()

with tabs[3]:
    st.subheader("Animation")
    anim_buffer = generate_animation(st.session_state.animation_params)
    if st.button("Generate Animation"):
        st.image(anim_buffer.getvalue(), format="GIF")

with tabs[4]:
    st.subheader("Gallery")
    if len(st.session_state.gallery_images) == 0:
        st.info("No images in gallery yet.")
    else:
        for idx, img in enumerate(st.session_state.gallery_images):
            st.image(img, caption=f"Gallery Image {idx+1}")

with tabs[5]:
    forum_ui()

with tabs[6]:
    settings_ui()

with tabs[7]:
    st.subheader("Export")
    st.info("Export options coming soon!")
