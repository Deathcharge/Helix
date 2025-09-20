import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
import io
import time
from PIL import Image

can_start, message = st.session_state.samsara_helix_context.can_start_new_session()
if not can_start:
    st.warning(message)
    st.stop()

# Attempt to import optional packages with error handling
try:
    import soundfile as sf
except ImportError:
    sf = None
    st.warning("`soundfile` not found. Audio synthesis will be disabled. Install with `pip install soundfile`.")

try:
    import imageio
except ImportError:
    imageio = None
    st.warning("`imageio` not found. Animation generation will be disabled. Install with `pip install imageio`.")

# Import your custom modules
from context_manager import SamsaraHelixContext
from agents import AGENTS
from ucf_protocol import format_ucf_message

# --- Page config ---
st.set_page_config(page_title="🕉️ Samsara Helix v∞ Limitless", layout="wide")

# --- Initialize session state ---
def init_session_state():
    defaults = {
        'chat_messages': [],
        'chat_input_value': "", # New key for chat input value
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
        },
        'rerun_triggered': False
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
    if sf is None:
        st.error("`soundfile` package is not installed. Please install it to use audio synthesis.")
        return None
    try:
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
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# --- Animation generation ---
def generate_animation(params):
    if imageio is None:
        st.error("`imageio` package is not installed. Please install it to use animation generation.")
        return None
    try:
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
                'colormap': 'hot', # Default for animation
                'color_invert': False,
                'fractal_type': 'mandelbrot' # Default for animation
            }
            img_array = generate_fractal(fractal_params)
            if img_array is None:
                return None # Propagate None if fractal generation fails
            frames.append(img_array)
        buffer = io.BytesIO()
        imageio.mimsave(buffer, frames, format='GIF', duration=0.1)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error generating animation: {e}")
        return None

# --- Chat command parser ---
def handle_chat_command(user_message: str):
    user_message_lower = user_message.lower()
    response = None

    if "generate mandelbrot" in user_message_lower:
        zoom_match = re.search(r"zoom(?:ed)? at (\d+\.?\d*)", user_message_lower) # Allow float zoom
        zoom = float(zoom_match.group(1)) if zoom_match else 1.0
        st.session_state.fractal_params.update({
            'zoom': zoom,
            'center_real': -0.7269,
            'center_imag': 0.1889,
            'iterations': 100,
            'width': 600,
            'height': 450,
            'fractal_type': "mandelbrot"
        })
        response = f"Generating Mandelbrot fractal with zoom level {zoom}x."
        st.session_state.rerun_triggered = True # Trigger rerun after command
    # Add more commands here as needed

    return response

# --- Chat UI ---
def chat_ui():
    st.header("🕉️ Samsara Helix Chat")

    # Display chat messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Callback function to handle sending message and clearing input
    def send_message_callback():
        user_input = st.session_state.chat_input_widget # Get value from the widget
        if not user_input:
            return

        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        st.session_state.chat_input_value = "" # Clear the value in session state

        command_response = handle_chat_command(user_input)
        if command_response:
            st.session_state.chat_messages.append({"role": "assistant", "content": command_response})
        else:
            # Generate AI response
            with st.spinner("Samsara Helix is thinking..."):
                ai_response = st.session_state.samsara_helix_context.generate_ucf_context(user_input, user_input)
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})

        st.session_state.rerun_triggered = True # Trigger rerun to update chat display

    # Input for new messages
    st.text_input(
        "Type your message here...",
        key="chat_input_widget", # Unique key for the widget
        value=st.session_state.chat_input_value, # Bind to session state value
        on_change=send_message_callback, # Call callback when Enter is pressed
        args=() # No arguments needed for on_change callback
    )
    st.button("Send", on_click=send_message_callback)


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
            if gif_buffer: # Check if gif_buffer is not None
                # Ensure the buffer is at the beginning for st.image
                gif_buffer.seek(0)
                st.image(gif_buffer.getvalue(), format="GIF", use_container_width=True) # Use getvalue() for BytesIO

# --- Gallery UI ---
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
# THIS LINE WAS MOVED UP TO FIX THE NAMERROR
tabs = st.tabs(["Fractal Studio", "Audio Synthesis", "Chat", "Animation", "Gallery", "Settings", "Export"])

with tabs[0]: # Fractal Studio
    st.subheader("Fractal Studio")
    st.write("Adjust fractal parameters and generate fractals.")

    params = st.session_state.fractal_params

    # Auto-generate checkbox moved outside the form
    auto_generate_changed = st.checkbox("Auto-generate fractal on parameter change", value=params['auto_generate'], key="auto_generate_fractal_checkbox")
    if auto_generate_changed != params['auto_generate']:
        params['auto_generate'] = auto_generate_changed
        st.session_state.rerun_triggered = True # Trigger rerun if auto-generate state changes

    # Use a form to group inputs and prevent immediate reruns on every slider change
    with st.form("fractal_form"):
        col1, col2 = st.columns(2)
        with col1:
            params['zoom'] = st.slider("Zoom", 0.0001, 2000.0, params['zoom'], step=0.001, format="%.4f", key="fractal_zoom")
            params['center_real'] = st.number_input("Center Real", value=params['center_real'], format="%.6f", key="fractal_center_real")
            params['iterations'] = st.slider("Iterations", 10, 1000, params['iterations'], key="fractal_iterations")
            params['fractal_type'] = st.selectbox("Fractal Type", ['mandelbrot', 'julia'], index=['mandelbrot', 'julia'].index(params['fractal_type']), key="fractal_type_select")
        with col2:
            params['center_imag'] = st.number_input("Center Imaginary", value=params['center_imag'], format="%.6f", key="fractal_center_imag")
            params['width'] = st.slider("Width (px)", 200, 1024, params['width'], key="fractal_width")
            params['height'] = st.slider("Height (px)", 200, 1024, params['height'], key="fractal_height")
            params['colormap'] = st.selectbox("Color Map", plt.colormaps(), index=plt.colormaps().index(params['colormap']), key="fractal_colormap")
            params['color_invert'] = st.checkbox("Invert Colors", value=params['color_invert'], key="fractal_color_invert")
            params['show_grid'] = st.checkbox("Show Grid Overlay (Future Feature)", value=params['show_grid'], disabled=True, key="fractal_show_grid")

        submitted = st.form_submit_button("Generate Fractal Image Now")

    # Display fractal if submitted or auto-generate is on
    if submitted or params['auto_generate']:
        with st.spinner("Generating fractal..."):
            img_array = generate_fractal(params)
            if img_array is not None:
                st.image(img_array, use_container_width=True)

with tabs[1]: # Audio Synthesis
    audio_ui()

with tabs[2]: # Chat
    chat_ui()

with tabs[3]: # Animation
    animation_ui()

with tabs[4]: # Gallery
    gallery_ui()

with tabs[5]: # Settings
    settings_ui()

with tabs[6]: # Export
    export_ui()

# --- Global rerun management ---
# Only rerun if explicitly triggered by an action that needs a full refresh
if st.session_state.get('rerun_triggered', False):
    st.session_state.rerun_triggered = False # Reset the flag
    st.rerun()
