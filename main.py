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

# =====================================
# CONFIGURATION
# =====================================
st.set_page_config(page_title="🕉️ Samsara Helix v∞", layout="wide")

FIREBASE_DB_URL = "https://project-helix-f77a1-default-rtdb.firebaseio.com/"

# =====================================
# FIREBASE INITIALIZATION
# =====================================
try:
    firebase_key_dict = json.loads(st.secrets["FIREBASE_KEY"])
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_key_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
except KeyError:
    st.error("❌ Firebase key missing. Add it in Streamlit Secrets.")
except Exception as e:
    st.error(f"⚠️ Firebase initialization error: {e}")

# =====================================
# SESSION STATE
# =====================================
def init_session_state():
    defaults = {
        'chat_messages': [],
        'fractal_params': {
            'zoom': 1.0,
            'center_real': -0.7269,
            'center_imag': 0.1889,
            'iterations': 200,
            'width': 600,
            'height': 450,
            'colormap': 'hot',
            'color_invert': False,
            'auto_generate': False,
        },
        'audio_params': {
            'base_frequency': 432,
            'duration_sec': 5,
            'volume': 0.5,
            'waveform': 'sine'
        },
        'animation_params': {
            'frame_count': 20,
            'width': 400,
            'height': 400,
            'zoom': 1.0,
            'center_real': -0.7269,
            'center_imag': 0.1889,
            'iterations': 150,
            'colormap': 'hot'
        },
        'gallery_images': [],
        'settings': {
            'theme': 'light',
            'language': 'English'
        },
        'rerun_triggered': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# =====================================
# FRACTAL GENERATION
# =====================================
def generate_fractal(params):
    try:
        w, h = params['width'], params['height']
        zoom = params['zoom']
        cr, ci = params['center_real'], params['center_imag']
        max_iter = params['iterations']

        scale = 3.0 / zoom
        x_min, x_max = cr - scale / 2, cr + scale / 2
        y_min, y_max = ci - scale / 2 * h / w, ci + scale / 2 * h / w

        x = np.linspace(x_min, x_max, w)
        y = np.linspace(y_min, y_max, h)
        C = x[np.newaxis, :] + 1j * y[:, np.newaxis]
        Z = np.zeros_like(C)
        escape_time = np.zeros(C.shape, dtype=float)

        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask] ** 2 + C[mask]
            escaped = (np.abs(Z) > 2) & (escape_time == 0)
            escape_time[escaped] = i + 1 - np.log2(np.log2(np.abs(Z[escaped])))

        escape_time[escape_time == 0] = max_iter
        norm = (escape_time - escape_time.min()) / (escape_time.max() - escape_time.min())
        if params.get('color_invert', False):
            norm = 1 - norm

        cmap = plt.get_cmap(params.get('colormap', 'hot'))
        img_array = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        return img_array
    except Exception as e:
        st.error(f"Fractal generation error: {e}")
        return None

# =====================================
# AUDIO GENERATION
# =====================================
def generate_audio(params):
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

        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, audio, fs, format='WAV')
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Audio generation error: {e}")
        return None

# =====================================
# ANIMATION GENERATION
# =====================================
def generate_animation(params):
    try:
        import imageio
        frames = []
        for i in range(params['frame_count']):
            local_params = params.copy()
            local_params['zoom'] = params['zoom'] * (1 + 0.05 * i)
            img_array = generate_fractal(local_params)
            frames.append(img_array)

        buffer = io.BytesIO()
        imageio.mimsave(buffer, frames, format='GIF', duration=0.1)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Animation error: {e}")
        return None

# =====================================
# FIREBASE FUNCTIONS - FORUM
# =====================================
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

def add_reply(thread_id, content, user):
    ref = db.reference(f"threads/{thread_id}/replies")
    new_reply = ref.push()
    new_reply.set({
        "content": content,
        "user": user,
        "timestamp": time.time()
    })

def get_threads():
    ref = db.reference("threads")
    return ref.get() or {}

# =====================================
# UI FUNCTIONS
# =====================================

# --- FRACTAL STUDIO ---
def fractal_ui():
    st.subheader("Fractal Studio")
    params = st.session_state.fractal_params

    params['zoom'] = st.slider("Zoom", 0.0001, 2000.0, params['zoom'], step=0.001)
    params['center_real'] = st.number_input("Center Real", value=params['center_real'])
    params['center_imag'] = st.number_input("Center Imaginary", value=params['center_imag'])
    params['iterations'] = st.slider("Iterations", 10, 1000, params['iterations'])
    params['colormap'] = st.selectbox("Color Map", plt.colormaps(), index=plt.colormaps().index(params['colormap']))
    params['color_invert'] = st.checkbox("Invert Colors", value=params['color_invert'])

    if st.button("Generate Fractal"):
        img = generate_fractal(params)
        if img is not None:
            st.image(img, use_container_width=True)
            st.session_state.gallery_images.append(img)

# --- AUDIO SYNTHESIS ---
def audio_ui():
    st.subheader("Audio Synthesis")
    params = st.session_state.audio_params
    params['base_frequency'] = st.slider("Base Frequency", 20, 2000, params['base_frequency'])
    params['duration_sec'] = st.slider("Duration", 1, 10, params['duration_sec'])
    params['volume'] = st.slider("Volume", 0.0, 1.0, params['volume'])
    params['waveform'] = st.selectbox("Waveform", ['sine', 'square', 'sawtooth'], index=['sine', 'square', 'sawtooth'].index(params['waveform']))

    if st.button("Generate Audio"):
        buffer = generate_audio(params)
        if buffer:
            st.audio(buffer, format="audio/wav")

# --- ANIMATION ---
def animation_ui():
    st.subheader("Animation")
    params = st.session_state.animation_params
    params['frame_count'] = st.slider("Frame Count", 10, 100, params['frame_count'])
    params['iterations'] = st.slider("Iterations", 50, 500, params['iterations'])

    if st.button("Generate Animation"):
        gif_buffer = generate_animation(params)
        if gif_buffer:
            st.image(gif_buffer.getvalue(), format="GIF", use_container_width=True)

# --- GALLERY ---
def gallery_ui():
    st.subheader("Gallery")
    if not st.session_state.gallery_images:
        st.info("No images yet.")
    else:
        for idx, img in enumerate(st.session_state.gallery_images):
            st.image(img, caption=f"Fractal {idx+1}")

# --- FORUM ---
def forum_ui():
    st.subheader("Community Forum 🧵")
    username = st.text_input("Your Username")

    st.markdown("### Create Thread")
    title = st.text_input("Thread Title")
    content = st.text_area("Thread Content")

    if st.button("Post Thread"):
        if username and title and content:
            create_thread(title, content, username)
            st.success("Thread posted!")
            st.rerun()
        else:
            st.error("Please complete all fields.")

    st.divider()
    st.markdown("### Recent Threads")
    threads = get_threads()
    if not threads:
        st.info("No threads yet.")
    else:
        for tid, data in sorted(threads.items(), key=lambda x: x[1]['timestamp'], reverse=True):
            st.markdown(f"#### {data['title']} (by {data['user']})")
            st.write(data['content'])
            st.caption(f"Posted: {time.ctime(data['timestamp'])}")

            reply = st.text_input(f"Reply to {tid}", key=f"reply-{tid}")
            if st.button(f"Reply-{tid}"):
                if reply and username:
                    add_reply(tid, reply, username)
                    st.rerun()

# --- SETTINGS ---
def settings_ui():
    st.subheader("Settings")
    st.write("Theme and Language preferences coming soon!")

# --- EXPORT ---
def export_ui():
    st.subheader("Export")
    if not st.session_state.gallery_images:
        st.info("No images to export.")
    else:
        img = Image.fromarray(st.session_state.gallery_images[-1])
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button("Download Last Fractal", buf.getvalue(), "fractal.png", "image/png")

# =====================================
# MAIN TAB LAYOUT
# =====================================
tabs = st.tabs([
    "Fractal Studio", "Audio Synthesis", "Animation",
    "Gallery", "Forum", "Settings", "Export"
])

with tabs[0]: fractal_ui()
with tabs[1]: audio_ui()
with tabs[2]: animation_ui()
with tabs[3]: gallery_ui()
with tabs[4]: forum_ui()
with tabs[5]: settings_ui()
with tabs[6]: export_ui()
