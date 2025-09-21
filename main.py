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

# ==============================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================
st.set_page_config(page_title="🕉️ Samsara Helix v∞", layout="wide")

# ==============================================
# FIREBASE INITIALIZATION (SAFE METHOD)
# ==============================================
FIREBASE_DB_URL = "https://project-helix-f77a1-default-rtdb.firebaseio.com/"

try:
    # Load JSON key directly from Streamlit secrets
    firebase_key_dict = json.loads(st.secrets["FIREBASE_KEY"])
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_key_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
except KeyError:
    st.error("❌ Firebase key not found in Streamlit secrets. Configure it under Settings → Secrets.")
except Exception as e:
    st.error(f"⚠️ Firebase initialization error: {e}")

# ==============================================
# OPTIONAL IMPORTS
# ==============================================
try:
    import soundfile as sf
except ImportError:
    sf = None
    st.warning("`soundfile` not found. Audio synthesis disabled. Install with `pip install soundfile`.")

try:
    import imageio
except ImportError:
    imageio = None
    st.warning("`imageio` not found. Animation generation disabled. Install with `pip install imageio`.")

# ==============================================
# FIREBASE FORUM FUNCTIONS
# ==============================================
def create_thread(title, content, user):
    """Create a new discussion thread in Firebase"""
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
    """Add a reply to a specific thread"""
    ref = db.reference(f"threads/{thread_id}/replies")
    new_reply = ref.push()
    new_reply.set({
        "content": reply_content,
        "user": user,
        "timestamp": time.time()
    })

def get_threads():
    """Retrieve all threads from Firebase"""
    ref = db.reference("threads")
    threads = ref.get()
    return threads or {}

# ==============================================
# SESSION STATE INITIALIZATION
# ==============================================
def init_session_state():
    defaults = {
        'chat_messages': [],
        'chat_input_value': "",
        'fractal_params': {
            'zoom': 1.0,
            'center_real': -0.7269,
            'center_imag': 0.1889,
            'iterations': 100,
            'width': 600,
            'height': 450,
            'auto_generate': False,
            'colormap': 'hot',
            'fractal_type': 'mandelbrot',
            'color_invert': False,
            'show_grid': False,
        },
        'audio_params': {
            'base_frequency': 432,
            'duration_sec': 5,
            'volume': 0.5,
            'waveform': 'sine'
        },
        'gallery_images': [],
        'settings': {
            'theme': 'light',
            'language': 'English',
            'auto_generate_fractal': False
        },
        'rerun_triggered': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ==============================================
# FRACTAL GENERATION
# ==============================================
def generate_fractal(params):
    try:
        width, height = params['width'], params['height']
        zoom = params['zoom']
        center_real = params['center_real']
        center_imag = params['center_imag']
        max_iter = params['iterations']
        colormap = params.get('colormap', 'hot')
        color_invert = params.get('color_invert', False)

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
        if color_invert:
            norm = 1 - norm
        cmap = plt.get_cmap(colormap)
        colored = cmap(norm)
        img_array = (colored[:, :, :3] * 255).astype(np.uint8)
        return img_array
    except Exception as e:
        st.error(f"Error generating fractal: {e}")
        return None

# ==============================================
# AUDIO GENERATION
# ==============================================
def generate_audio(params):
    if sf is None:
        st.error("`soundfile` is required for audio synthesis.")
        return None
    try:
        fs = 44100
        t = np.linspace(0, params['duration_sec'], int(fs * params['duration_sec']), False)
        freq = params['base_frequency']
        waveform = params['waveform']

        if waveform == 'sine':
            audio = np.sin(2 * np.pi * freq * t)
        elif waveform == 'square':
            audio = np.sign(np.sin(2 * np.pi * freq * t))
        elif waveform == 'sawtooth':
            audio = 2 * (t * freq - np.floor(0.5 + t * freq))
        else:
            audio = np.sin(2 * np.pi * freq * t)

        audio *= params['volume']
        audio = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        sf.write(buffer, audio, fs, format='WAV')
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# ==============================================
# FORUM INTERFACE
# ==============================================
def forum_ui():
    st.header("Community Forum 🧵")
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
    else:
        sorted_threads = sorted(threads.items(), key=lambda x: x[1]['timestamp'], reverse=True)
        for thread_id, thread_data in sorted_threads:
            st.markdown(f"### {thread_data['title']} (by {thread_data['user']})")
            st.write(thread_data['content'])
            st.caption(f"Posted: {time.ctime(thread_data['timestamp'])}")

            if "replies" in thread_data and thread_data["replies"]:
                st.markdown("**Replies:**")
                for reply_id, reply in thread_data["replies"].items():
                    st.write(f"- {reply['user']}: {reply['content']}")

            reply_content = st.text_input(f"Reply to {thread_id}", key=f"reply-{thread_id}")
            if st.button(f"Reply-{thread_id}"):
                if reply_content and username:
                    add_reply(thread_id, reply_content, username)
                    st.success("Reply added!")
                    st.rerun()

# ==============================================
# GALLERY, SETTINGS, AND EXPORT
# ==============================================
def gallery_ui():
    st.header("🖼️ Gallery")
    if len(st.session_state.gallery_images) == 0:
        st.info("No images in gallery yet.")
    else:
        for idx, img in enumerate(st.session_state.gallery_images):
            st.image(img, caption=f"Gallery Image {idx+1}", use_container_width=True)

    if st.button("Add Current Fractal to Gallery"):
        img_array = generate_fractal(st.session_state.fractal_params)
        if img_array is not None:
            st.session_state.gallery_images.append(img_array)
            st.success("Added current fractal to gallery!")

def settings_ui():
    st.header("⚙️ Settings")
    settings = st.session_state.settings
    settings['theme'] = st.selectbox("Theme", ['light', 'dark'], index=['light', 'dark'].index(settings['theme']))
    settings['language'] = st.selectbox("Language", ['English', 'Sanskrit', 'Other'],
                                        index=['English', 'Sanskrit', 'Other'].index(settings['language']))
    settings['auto_generate_fractal'] = st.checkbox("Auto-generate fractal on parameter change",
                                                    value=settings['auto_generate_fractal'])

def export_ui():
    st.header("📤 Export")
    if len(st.session_state.gallery_images) == 0:
        st.info("No images to export.")
    else:
        img_array = st.session_state.gallery_images[-1]
        img = Image.fromarray(img_array)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Last Gallery Image as PNG",
            data=byte_im,
            file_name="samsara_helix_fractal.png",
            mime="image/png"
        )

# ==============================================
# APP TAB STRUCTURE
# ==============================================
tabs = st.tabs([
    "Fractal Studio", "Audio Synthesis", "Gallery", "Forum", "Settings", "Export"
])

with tabs[0]:
    st.subheader("Fractal Studio")
    params = st.session_state.fractal_params
    auto_generate_changed = st.checkbox("Auto-generate fractal", value=params['auto_generate'])
    if auto_generate_changed != params['auto_generate']:
        params['auto_generate'] = auto_generate_changed
        st.session_state.rerun_triggered = True

    with st.form("fractal_form"):
        params['zoom'] = st.slider("Zoom", 0.0001, 2000.0, params['zoom'], step=0.001)
        params['iterations'] = st.slider("Iterations", 10, 1000, params['iterations'])
        params['colormap'] = st.selectbox("Color Map", plt.colormaps(),
                                          index=plt.colormaps().index(params['colormap']))
        submitted = st.form_submit_button("Generate")

    if submitted or params['auto_generate']:
        img_array = generate_fractal(params)
        if img_array is not None:
            st.image(img_array, use_container_width=True)

with tabs[1]:
    st.subheader("Audio Synthesis")
    params = st.session_state.audio_params
    params['base_frequency'] = st.slider("Base Frequency (Hz)", 20, 2000, params['base_frequency'])
    params['duration_sec'] = st.slider("Duration (seconds)", 1, 10, params['duration_sec'])
    params['volume'] = st.slider("Volume", 0.0, 1.0, params['volume'])
    params['waveform'] = st.selectbox("Waveform", ['sine', 'square', 'sawtooth'],
                                      index=['sine', 'square', 'sawtooth'].index(params['waveform']))

    if st.button("Generate Audio"):
        audio_buffer = generate_audio(params)
        if audio_buffer:
            st.audio(audio_buffer, format='audio/wav')

with tabs[2]:
    gallery_ui()

with tabs[3]:
    forum_ui()

with tabs[4]:
    settings_ui()

with tabs[5]:
    export_ui()
