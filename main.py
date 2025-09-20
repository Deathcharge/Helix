import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import time
from PIL import Image
from context_manager import SamsaraHelixContext
from agents import AGENTS
from ucf_protocol import format_ucf_message

# --- Page config ---
st.set_page_config(page_title="🕉️ Samsara Helix v∞ Limitless", layout="wide")

# --- Initialize session state ---
def init_session_state():
    defaults = {
        'chat_messages': [],
        'chat_input': "",
        'rerun_flag': False,
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
            'show_grid': False,
            'color_invert': False,
        },
        'audio_params': {
            'base_frequency': 432,
            'duration_sec': 5,
            'volume': 0.5,
            'waveform': 'sine'
        },
        'animation_params': {
            'frame_count': 30,
            'width': 400,
            'height': 400,
            'zoom': 1.0,
            'center_real': -0.7269,
            'center_imag': 0.1889,
            'iterations': 100
        },
        'gallery_images': [],
        'settings': {
            'theme': 'light',
            'language': 'English',
            'auto_generate_fractal': False
        }
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# Initialize Samsara Helix Context
if 'samsara_helix_context' not in st.session_state:
    st.session_state.samsara_helix_context = SamsaraHelixContext()

# --- CSS for chat bubbles ---
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 10px;
    background: #f9f9f9;
}
.user-msg {
    background-color: #4A90E2;
    color: white;
    padding: 10px;
    border-radius: 15px;
    margin: 5px 0;
    max-width: 80%;
    align-self: flex-end;
    word-wrap: break-word;
    font-size: 1rem;
}
.assistant-msg {
    background-color: #e1e1e1;
    color: black;
    padding: 10px;
    border-radius: 15px;
    margin: 5px 0;
    max-width: 80%;
    align-self: flex-start;
    word-wrap: break-word;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

# --- Tab layout ---
tabs = st.tabs(["Fractal Studio", "Audio Synthesis", "Chat", "Animation", "Gallery", "Settings", "Export"])

# --- Fractal generation ---
def generate_fractal(params):
    try:
        width, height = params['width'], params['height']
        zoom = params['zoom']
        center_real = params['center_real']
        center_imag = params['center_imag']
        max_iter = params['iterations']
        fractal_type = params.get('fractal_type', 'mandelbrot')
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

# --- Audio synthesis ---
def generate_audio(params):
    try:
        import numpy as np
        import soundfile as sf
        import io

        fs = 44100  # Sample rate
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
    except ImportError:
        st.error("Please install the 'soundfile' package to enable audio synthesis: `pip install soundfile`")
        return None
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# --- Animation generation ---
def generate_animation(params):
    try:
        import imageio
        frames = []
        for i in range(params['frame_count']):
            zoom = params['zoom'] * (1 + i / params['frame_count'])
            fractal_params = {
                'zoom': zoom,
                'center_real': params['center_real'],
                'center_imag': params['center_imag'],
                'iterations': params['iterations'],
                'width': params['width'],
                'height': params['height'],
                'colormap': 'hot',
                'color_invert': False,
                'fractal_type': 'mandelbrot'
            }
            img_array = generate_fractal(fractal_params)
            if img_array is None:
                return None
            frames.append(img_array)
        buffer = io.BytesIO()
        imageio.mimsave(buffer, frames, format='GIF', duration=0.1)
        buffer.seek(0)
        return buffer
    except ImportError:
        st.error("Please install the 'imageio' package to enable animation generation: `pip install imageio`")
        return None
    except Exception as e:
        st.error(f"Error generating animation: {e}")
        return None

# --- Chat command parser ---
def handle_chat_command(user_message: str):
    user_message_lower = user_message.lower()
    response = None

    if "generate mandelbrot" in user_message_lower:
        zoom_match = re.search(r"zoom(?:ed)? at (\d+)", user_message_lower)
        zoom = float(zoom_match.group(1)) if zoom_match else 1.0
        st.session_state.fractal_params.update({
            'zoom': zoom,
            'center_real': -0.7269,
            'center_imag': 0.1889,
            'iterations': 100,
            'width': 600,
            'height': 450
        })
        st.session_state.fractal_params['fractal_type'] = "mandelbrot"
        st.session_state.rerun_flag = True
        response = f"Generating Mandelbrot fractal with zoom level {zoom}x."
    # Add more commands here as needed

    return response

# --- Chat UI ---
def chat_ui():
    st.header("🕉️ Samsara Helix Chat")

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    def send_chat_message():
        user_msg = st.session_state.chat_input.strip()
        if not user_msg:
            return

        st.session_state.chat_messages.append({"role": "user", "content": user_msg})

        command_response = handle_chat_command(user_msg)
        if command_response:
            st.session_state.chat_messages.append({"role": "assistant", "content": command_response})
        else:
            ai_response = st.session_state.samsara_helix_context.generate_ucf_context(user_msg, user_msg)
            st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})

        st.session_state.chat_input = ""
        st.session_state.rerun_flag = True

    st.text_input("Type your message here...", key="chat_input", on_change=send_chat_message)
    st.button("Send", on_click=send_chat_message)

# --- Audio Synthesis UI ---
def audio_ui():
    st.header("🎵 Audio Synthesis")

    params = st.session_state.audio_params
    params['base_frequency'] = st.slider("Base Frequency (Hz)", 20, 2000, params['base_frequency'])
    params['duration_sec'] = st.slider("Duration (seconds)", 1, 10, params['duration_sec'])
    params['volume'] = st.slider("Volume", 0.0, 1.0, params['volume'])
    params['waveform'] = st.selectbox("Waveform", ['sine', 'square', 'sawtooth'], index=['sine', 'square', 'sawtooth'].index(params['waveform']))

    if st.button("Generate Audio"):
        audio_buffer = generate_audio(params)
        if audio_buffer:
            st.audio(audio_buffer, format='audio/wav')

# --- Animation UI ---
def animation_ui():
    st.header("🎞️ Animation")

    params = st.session_state.animation_params
    params['frame_count'] = st.slider("Frame Count", 10, 100, params['frame_count'])
    params['width'] = st.slider("Width (px)", 200, 800, params['width'])
    params['height'] = st.slider("Height (px)", 200, 800, params['height'])
    params['zoom'] = st.slider("Zoom", 1.0, 10.0, params['zoom'])
    params['center_real'] = st.number_input("Center Real", value=params['center_real'])
    params['center_imag'] = st.number_input("Center Imaginary", value=params['center_imag'])
    params['iterations'] = st.slider("Iterations", 50, 500, params['iterations'])

    if st.button("Generate Animation"):
        with st.spinner("Generating animation..."):
            gif_buffer = generate_animation(params)
            if gif_buffer:
                st.image(gif_buffer, format="GIF")

# --- Gallery UI ---
def gallery_ui():
    st.header("🖼️ Gallery")

    if len(st.session_state.gallery_images) == 0:
        st.info("No images in gallery yet.")
    else:
        for idx, img in enumerate(st.session_state.gallery_images):
            st.image(img, caption=f"Gallery Image {idx+1}")

    if st.button("Add Current Fractal to Gallery"):
        img_array = generate_fractal(st.session_state.fractal_params)
        if img_array is not None:
            st.session_state.gallery_images.append(img_array)
            st.success("Added current fractal to gallery!")

# --- Settings UI ---
def settings_ui():
    st.header("⚙️ Settings")

    settings = st.session_state.settings
    settings['theme'] = st.selectbox("Theme", ['light', 'dark'], index=['light', 'dark'].index(settings['theme']))
    settings['language'] = st.selectbox("Language", ['English', 'Sanskrit', 'Other'], index=['English', 'Sanskrit', 'Other'].index(settings['language']))
    settings['auto_generate_fractal'] = st.checkbox("Auto-generate fractal on parameter change", value=settings['auto_generate_fractal'])

# --- Export UI ---
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

# --- Render tabs ---
with tabs[0]:
    st.subheader("Fractal Studio")
    st.write("Adjust fractal parameters and generate fractals.")

    params = st.session_state.fractal_params

    col1, col2 = st.columns(2)
    with col1:
        params['zoom'] = st.slider("Zoom", 0.1, 20.0, params['zoom'], step=0.1)
        params['center_real'] = st.number_input("Center Real", value=params['center_real'], format="%.6f")
        params['iterations'] = st.slider("Iterations", 10, 1000, params['iterations'])
        params['fractal_type'] = st.selectbox("Fractal Type", ['mandelbrot', 'julia'], index=['mandelbrot', 'julia'].index(params['fractal_type']))
    with col2:
        params['center_imag'] = st.number_input("Center Imaginary", value=params['center_imag'], format="%.6f")
        params['width'] = st.slider("Width (px)", 200, 1200, params['width'])
        params['height'] = st.slider("Height (px)", 200, 1200, params['height'])
        params['colormap'] = st.selectbox("Color Map", plt.colormaps(), index=plt.colormaps().index(params['colormap']))
        params['color_invert'] = st.checkbox("Invert Colors", value=params.get('color_invert', False))
        params['show_grid'] = st.checkbox("Show Grid Overlay", value=params.get('show_grid', False))

    if st.button("Generate Fractal Image") or (params.get('auto_generate', False) and st.session_state.rerun_flag):
        with st.spinner("Generating fractal..."):
            img_array = generate_fractal(params)
            if img_array is not None:
                st.image(img_array, use_column_width=True)
                if params['show_grid']:
                    st.markdown("<small>Grid overlay feature coming soon.</small>", unsafe_allow_html=True)

with tabs[1]:
    audio_ui()

with tabs[2]:
    chat_ui()

with tabs[3]:
    animation_ui()

with tabs[4]:
    gallery_ui()

with tabs[5]:
    settings_ui()

with tabs[6]:
    export_ui()

# --- Handle rerun flag ---
if st.session_state.get('rerun_flag', False):
    st.session_state.rerun_flag = False
    st.experimental_rerun()
