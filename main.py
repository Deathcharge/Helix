import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import time
import json
import re
from PIL import Image

# -------------------------------
# Optional Imports (graceful fallback)
# -------------------------------
try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import imageio
except ImportError:
    imageio = None

# -------------------------------
# Third-party APIs
# -------------------------------
import firebase_admin
from firebase_admin import credentials, db

import openai

# -------------------------------
# Internal modules (repo files)
# -------------------------------
# These are expected to exist in your repo.
from context_manager import SamsaraHelixContext
from agents import AGENTS  # noqa: F401  (imported for context_manager)
from ucf_protocol import format_ucf_message  # noqa: F401


# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="🕉️ Samsara Helix v∞", layout="wide")

# -------------------------------
# Secrets / Keys
# -------------------------------
# OpenAI
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
if OPENAI_KEY:
    try:
        openai.api_key = OPENAI_KEY  # compatible with openai>=1.0 (chat.completions below)
    except Exception as e:
        st.warning(f"OpenAI key present but could not be applied: {e}")
else:
    st.info("ℹ️ No OPENAI_API_KEY in secrets — Chat will show a friendly warning instead of erroring.")

# Firebase
FIREBASE_DB_URL = "https://project-helix-f77a1-default-rtdb.firebaseio.com/"
_firebase_ready = False
try:
    fb_json_str = st.secrets.get("firebase_service_key", "")
    if fb_json_str:
        firebase_key_dict = json.loads(fb_json_str)
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_key_dict)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
        _firebase_ready = True
    else:
        st.info("ℹ️ No firebase_service_key in secrets — Forum tab will show a setup hint.")
except Exception as e:
    st.error(f"⚠️ Firebase initialization error: {e}")


# -------------------------------
# Session State Defaults
# -------------------------------
def init_session_state():
    defaults = {
        "chat_messages": [],  # [{"role": "user"/"assistant","content": "..."}]
        "chat_input_value": "",
        "fractal_params": {
            "zoom": 1.0,
            "center_real": -0.7269,
            "center_imag": 0.1889,
            "iterations": 100,
            "width": 600,
            "height": 450,
            "auto_generate": False,
            "colormap": "hot",
            "fractal_type": "mandelbrot",
            "color_invert": False,
            "show_grid": False,
        },
        "audio_params": {
            "base_frequency": 432,
            "duration_sec": 5,
            "volume": 0.5,
            "waveform": "sine",  # sine | square | sawtooth
        },
        "animation_params": {
            "frame_count": 30,
            "width": 400,
            "height": 400,
            "zoom": 1.0,
            "center_real": -0.7269,
            "center_imag": 0.1889,
            "iterations": 100,
            "colormap": "hot",
            "color_invert": False,
        },
        "gallery_images": [],           # list[np.ndarray]
        "last_animation_gif": None,     # bytes (GIF) or None
        "settings": {
            "theme": "light",
            "language": "English",
            "auto_generate_fractal": False,
        },
        "rerun_triggered": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

if "samsara_helix_context" not in st.session_state:
    st.session_state.samsara_helix_context = SamsaraHelixContext()


# -------------------------------
# Chat UI styles
# -------------------------------
st.markdown(
    """
<style>
.chat-container {
  display: flex; flex-direction: column;
  max-height: 420px; overflow-y: auto;
  padding: 10px; border: 1px solid #ddd;
  border-radius: 12px; background: #fafafa;
}
.user-msg {
  background-color: #4A90E2; color: #fff;
  padding: 10px 14px; border-radius: 16px;
  margin: 6px 0; max-width: 82%; align-self: flex-end;
  word-wrap: break-word; font-size: 0.98rem;
}
.assistant-msg {
  background-color: #e9e9e9; color: #222;
  padding: 10px 14px; border-radius: 16px;
  margin: 6px 0; max-width: 82%; align-self: flex-start;
  word-wrap: break-word; font-size: 0.98rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------------
# Utilities
# -------------------------------
def _safe_colormap(name: str) -> str:
    # Return the same name if valid; otherwise fall back.
    try:
        _ = plt.get_cmap(name)
        return name
    except Exception:
        return "hot"


# -------------------------------
# Fractal Generation
# -------------------------------
def generate_fractal(params: dict):
    try:
        width, height = int(params["width"]), int(params["height"])
        zoom = float(params["zoom"])
        cx = float(params["center_real"])
        cy = float(params["center_imag"])
        max_iter = int(params["iterations"])
        colormap = _safe_colormap(params.get("colormap", "hot"))
        invert = bool(params.get("color_invert", False))

        # viewport
        scale = 3.0 / max(zoom, 1e-9)
        x_min = cx - scale / 2
        x_max = cx + scale / 2
        y_min = cy - scale / 2 * height / width
        y_max = cy + scale / 2 * height / width

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
            if escaped.any():
                escape_time[escaped] = i + 1 - np.log2(np.log2(np.abs(Z[escaped])))

        escape_time[escape_time == 0] = max_iter
        norm = (escape_time - escape_time.min()) / max(1e-9, (escape_time.max() - escape_time.min()))
        if invert:
            norm = 1 - norm

        cmap = plt.get_cmap(colormap)
        colored = cmap(norm)
        img = (colored[:, :, :3] * 255).astype(np.uint8)
        return img
    except Exception as e:
        st.error(f"Error generating fractal: {e}")
        return None


# -------------------------------
# Audio Synthesis
# -------------------------------
def generate_audio(params: dict):
    if sf is None:
        st.warning("⚠️ Install `soundfile` to enable audio synthesis: `pip install soundfile`")
        return None
    try:
        fs = 44100
        t = np.linspace(0, params["duration_sec"], int(fs * params["duration_sec"]), False)
        freq = float(params["base_frequency"])
        wave = params["waveform"]

        if wave == "sine":
            audio = np.sin(2 * np.pi * freq * t)
        elif wave == "square":
            audio = np.sign(np.sin(2 * np.pi * freq * t))
        elif wave == "sawtooth":
            audio = 2 * (t * freq - np.floor(0.5 + t * freq))
        else:
            audio = np.sin(2 * np.pi * freq * t)

        audio = (audio * float(params["volume"])).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, audio, fs, format="WAV")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        return None


# -------------------------------
# Animation (GIF)
# -------------------------------
def generate_animation(params: dict):
    if imageio is None:
        st.warning("⚠️ Install `imageio` to enable animations: `pip install imageio`")
        return None
    try:
        frames = []
        frame_count = int(params["frame_count"])
        for i in range(frame_count):
            p = dict(params)
            p["zoom"] = float(params["zoom"]) * (1 + (i / max(1, frame_count)))
            img = generate_fractal(p)
            if img is None:
                return None
            frames.append(img)

        gif_buf = io.BytesIO()
        imageio.mimsave(gif_buf, frames, format="GIF", duration=0.1)
        gif_buf.seek(0)
        return gif_buf
    except Exception as e:
        st.error(f"Animation generation failed: {e}")
        return None


# -------------------------------
# Forum (Firebase)
# -------------------------------
def _ensure_firebase_ready():
    if not _firebase_ready:
        st.warning("⚠️ Firebase is not configured. Add `firebase_service_key` to Secrets to enable the forum.")
        return False
    return True


def create_thread(title: str, content: str, user: str):
    if not _ensure_firebase_ready():
        return
    ref = db.reference("threads")
    new_thread = ref.push()
    new_thread.set({
        "title": title,
        "content": content,
        "user": user,
        "timestamp": time.time(),
        "replies": {}
    })


def add_reply(thread_id: str, reply_content: str, user: str):
    if not _ensure_firebase_ready():
        return
    ref = db.reference(f"threads/{thread_id}/replies")
    new_reply = ref.push()
    new_reply.set({
        "content": reply_content,
        "user": user,
        "timestamp": time.time()
    })


def get_threads():
    if not _ensure_firebase_ready():
        return {}
    ref = db.reference("threads")
    threads = ref.get()
    return threads or {}


def forum_ui():
    st.header("Community Forum 🧵")
    if not _firebase_ready:
        st.info("Add your Firebase service account JSON to **Secrets** as `firebase_service_key` to enable this tab.")
        return

    username = st.text_input("Your Username")

    st.subheader("Create New Thread")
    title = st.text_input("Thread Title")
    content = st.text_area("Thread Content")

    if st.button("Post Thread"):
        if title and content and username:
            create_thread(title, content, username)
            st.success("Thread posted successfully!")
            st.rerun()
        else:
            st.error("Please provide title, content, and username.")

    st.divider()
    st.subheader("🔥 Recent Threads")

    threads = get_threads()
    if not threads:
        st.info("No threads yet. Be the first to start a discussion!")
        return

    # newest first
    sorted_threads = sorted(threads.items(), key=lambda x: x[1].get("timestamp", 0), reverse=True)
    for thread_id, thread_data in sorted_threads:
        st.markdown(f"### {thread_data.get('title','(untitled)')} (by {thread_data.get('user','anon')})")
        st.write(thread_data.get("content", ""))
        ts = thread_data.get("timestamp", 0)
        if ts:
            st.caption(f"Posted: {time.ctime(ts)}")

        # replies
        replies = thread_data.get("replies", {})
        if replies:
            st.markdown("**Replies:**")
            for _, reply in replies.items():
                st.write(f"- **{reply.get('user','anon')}:** {reply.get('content','')}")
        else:
            st.caption("No replies yet.")

        reply_content = st.text_input(f"Reply to {thread_id}", key=f"reply-{thread_id}")
        if st.button(f"Reply-{thread_id}"):
            if reply_content and username:
                add_reply(thread_id, reply_content, username)
                st.success("Reply added!")
                st.rerun()
            else:
                st.error("Please enter a reply and your username.")


# -------------------------------
# Chat logic
# -------------------------------
def handle_chat_command(user_message: str):
    msg = user_message.lower()
    # Example command: "generate mandelbrot zoomed at 2.5"
    if "generate mandelbrot" in msg:
        zoom_match = re.search(r"zoom(?:ed)?(?: at)? (\d+(?:\.\d+)?)", msg)
        zoom = float(zoom_match.group(1)) if zoom_match else 1.0
        st.session_state.fractal_params.update({
            "zoom": zoom,
            "center_real": -0.7269,
            "center_imag": 0.1889,
            "iterations": 100,
            "width": 600,
            "height": 450,
            "fractal_type": "mandelbrot",
        })
        st.session_state.rerun_triggered = True
        return f"🔮 Generating Mandelbrot fractal at {zoom}× zoom."
    return None


def chat_ui():
    st.header("🕉️ Samsara Helix Chat")

    # Chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_messages:
        cls = "user-msg" if msg["role"] == "user" else "assistant-msg"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    def _send():
        text = st.session_state.get("chat_input_widget", "").strip()
        if not text:
            return
        st.session_state.chat_messages.append({"role": "user", "content": text})

        # Commands (update UI)
        cmd = handle_chat_command(text)
        if cmd:
            st.session_state.chat_messages.append({"role": "assistant", "content": cmd})
        else:
            # AI response
            if not OPENAI_KEY:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": "⚠️ OpenAI key missing in secrets. Add `OPENAI_API_KEY` to enable chat responses."
                })
            else:
                with st.spinner("Samsara Helix is thinking..."):
                    # Delegate to your context manager (uses OpenAI under the hood)
                    try:
                        reply = st.session_state.samsara_helix_context.generate_ucf_context(text, text)
                    except Exception as e:
                        reply = f"⚠️ Chat error: {e}"
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})

        # Clear
        st.session_state["chat_input_widget"] = ""
        st.session_state.rerun_triggered = True

    st.text_input(
        "Type your message...",
        key="chat_input_widget",
        value=st.session_state.get("chat_input_value", ""),
        on_change=_send,
    )
    st.button("Send", on_click=_send)


# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs([
    "Fractal Studio",
    "Audio Synthesis",
    "Chat",
    "Animation",
    "Gallery",
    "Forum",
    "Settings",
    "Export",
])

# Fractal Studio
with tabs[0]:
    st.subheader("Fractal Studio")
    st.write("Adjust parameters and generate beautiful fractals.")

    params = st.session_state.fractal_params

    auto_generate_changed = st.checkbox(
        "Auto-generate fractal on parameter change",
        value=params["auto_generate"],
        key="auto_generate_fractal_checkbox",
    )
    if auto_generate_changed != params["auto_generate"]:
        params["auto_generate"] = auto_generate_changed
        st.session_state.rerun_triggered = True

    with st.form("fractal_form"):
        col1, col2 = st.columns(2)
        with col1:
            params["zoom"] = st.slider("Zoom", 0.0001, 2000.0, params["zoom"], step=0.001, format="%.4f")
            params["center_real"] = st.number_input("Center Real", value=params["center_real"], format="%.6f")
            params["iterations"] = st.slider("Iterations", 10, 1000, params["iterations"])
            params["fractal_type"] = st.selectbox("Fractal Type", ["mandelbrot", "julia"], index=0)
        with col2:
            params["center_imag"] = st.number_input("Center Imaginary", value=params["center_imag"], format="%.6f")
            params["width"] = st.slider("Width (px)", 200, 1024, params["width"])
            params["height"] = st.slider("Height (px)", 200, 1024, params["height"])
            cmap_names = list(plt.colormaps())
            idx = cmap_names.index(params["colormap"]) if params["colormap"] in cmap_names else cmap_names.index("hot")
            params["colormap"] = st.selectbox("Color Map", cmap_names, index=idx)
            params["color_invert"] = st.checkbox("Invert Colors", value=params["color_invert"])

        submitted = st.form_submit_button("Generate Fractal Image Now")

    if submitted or params["auto_generate"]:
        with st.spinner("Generating fractal..."):
            img_array = generate_fractal(params)
            if img_array is not None:
                st.image(img_array, use_container_width=True)

    if st.button("Add Current Fractal to Gallery"):
        img_array = generate_fractal(params)
        if img_array is not None:
            st.session_state.gallery_images.append(img_array)
            st.success("Added to gallery!")

# Audio Synthesis
with tabs[1]:
    st.subheader("🎵 Audio Synthesis")
    a = st.session_state.audio_params
    a["base_frequency"] = st.slider("Base Frequency (Hz)", 20, 2000, a["base_frequency"])
    a["duration_sec"] = st.slider("Duration (seconds)", 1, 10, a["duration_sec"])
    a["volume"] = st.slider("Volume", 0.0, 1.0, a["volume"])
    a["waveform"] = st.selectbox("Waveform", ["sine", "square", "sawtooth"], index=["sine", "square", "sawtooth"].index(a["waveform"]))
    if st.button("Generate Audio"):
        buf = generate_audio(a)
        if buf:
            st.audio(buf, format="audio/wav")

# Chat
with tabs[2]:
    chat_ui()

# Animation
with tabs[3]:
    st.subheader("🎞️ Animation (Fractal Zoom)")
    an = st.session_state.animation_params
    an["frame_count"] = st.slider("Frame Count", 10, 120, an["frame_count"])
    an["width"] = st.slider("Width (px)", 200, 1024, an["width"])
    an["height"] = st.slider("Height (px)", 200, 1024, an["height"])
    an["zoom"] = st.slider("Base Zoom", 0.1, 20.0, float(an["zoom"]))
    an["center_real"] = st.number_input("Center Real", value=float(an["center_real"]))
    an["center_imag"] = st.number_input("Center Imaginary", value=float(an["center_imag"]))
    an["iterations"] = st.slider("Iterations", 10, 1000, int(an["iterations"]))
    cmap_names = list(plt.colormaps())
    idx = cmap_names.index(an["colormap"]) if an["colormap"] in cmap_names else cmap_names.index("hot")
    an["colormap"] = st.selectbox("Color Map", cmap_names, index=idx)
    an["color_invert"] = st.checkbox("Invert Colors", value=bool(an["color_invert"]))

    if st.button("Generate GIF"):
        with st.spinner("Rendering animation..."):
            gif_buf = generate_animation(an)
            if gif_buf:
                st.image(gif_buf.getvalue(), format="GIF", use_container_width=True)
                st.session_state.last_animation_gif = gif_buf.getvalue()
                st.success("Animation generated!")

# Gallery
with tabs[4]:
    st.subheader("🖼️ Gallery")
    if len(st.session_state.gallery_images) == 0:
        st.info("No images in gallery yet.")
    else:
        for i, img in enumerate(st.session_state.gallery_images):
            st.image(img, caption=f"Gallery Image {i+1}", use_container_width=True)

# Forum
with tabs[5]:
    forum_ui()

# Settings
with tabs[6]:
    st.subheader("⚙️ Settings")
    s = st.session_state.settings
    s["theme"] = st.selectbox("Theme", ["light", "dark"], index=["light", "dark"].index(s["theme"]))
    s["language"] = st.selectbox("Language", ["English", "Sanskrit", "Other"], index=["English", "Sanskrit", "Other"].index(s["language"]))
    s["auto_generate_fractal"] = st.checkbox("Auto-generate fractal in Fractal Studio", value=s["auto_generate_fractal"])

# Export
with tabs[7]:
    st.subheader("📤 Export")
    if st.session_state.gallery_images:
        img = Image.fromarray(st.session_state.gallery_images[-1])
        png_buf = io.BytesIO()
        img.save(png_buf, format="PNG")
        st.download_button("Download Last Fractal (PNG)", data=png_buf.getvalue(), file_name="fractal.png", mime="image/png")
    else:
        st.info("No fractal images to export yet.")

    if st.session_state.last_animation_gif:
        st.download_button("Download Last Animation (GIF)", data=st.session_state.last_animation_gif, file_name="fractal_zoom.gif", mime="image/gif")
    else:
        st.caption("Generate an animation to enable GIF download.")

# Rerun management for chat/auto updates
if st.session_state.get("rerun_triggered", False):
    st.session_state.rerun_triggered = False
    st.experimental_rerun()
