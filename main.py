import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
import time
import re
import random

from context_manager import SamsaraHelixContext
from agents import AGENTS
from ucf_protocol import format_ucf_message

# --- Page config ---
st.set_page_config(page_title="🕉️ Samsara Helix v∞ Limitless", layout="wide")

# --- Initialize session state ---
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""
if 'rerun_flag' not in st.session_state:
    st.session_state.rerun_flag = False
if 'fractal_params' not in st.session_state:
    st.session_state.fractal_params = {
        'zoom': 1.0,
        'center_real': -0.7269,
        'center_imag': 0.1889,
        'iterations': 100,
        'width': 600,
        'height': 450
    }
if 'current_fractal_type' not in st.session_state:
    st.session_state.current_fractal_type = "mandelbrot"
if 'current_colormap' not in st.session_state:
    st.session_state.current_colormap = "hot"
if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False

if 'samsara_helix_context' not in st.session_state:
    st.session_state.samsara_helix_context = SamsaraHelixContext()

# --- CSS for chat bubbles and UI ---
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
.stButton > button {
    background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
    background-size: 300% 300%;
    animation: gradient-shift 4s ease infinite, pulse-glow 2s ease-in-out infinite alternate;
    font-size: 1.1rem;
    height: 3rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    box-shadow: 0 8px 32px rgba(255, 107, 107, 0.4);
    border-radius: 16px;
    border: none;
    color: white;
    cursor: pointer;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 12px 48px rgba(255, 107, 107, 0.6);
}
@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulse-glow {
    0% {
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.4), 0 0 0 0 rgba(255, 107, 107, 0.4);
    }
    100% {
        box-shadow: 0 12px 48px rgba(255, 107, 107, 0.6), 0 0 0 8px rgba(255, 107, 107, 0);
    }
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar navigation ---
page = st.sidebar.radio("Navigate", ["Chat", "Fractal Studio", "Audio Synthesis", "Animation", "Gallery", "Settings", "Export"])

# --- Fractal generation function ---
def generate_fractal(params):
    width, height = params['width'], params['height']
    zoom = params['zoom']
    center_real = params['center_real']
    center_imag = params['center_imag']
    max_iter = params['iterations']
    fractal_type = st.session_state.current_fractal_type

    # Mandelbrot example implementation
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
    cmap = plt.get_cmap(st.session_state.current_colormap)
    colored = cmap(norm)
    img_array = (colored[:, :, :3] * 255).astype(np.uint8)
    return img_array

# --- Chat command parser ---
def handle_chat_command(user_message: str):
    user_message_lower = user_message.lower()
    response = None

    # Mandelbrot generation command
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
        st.session_state.current_fractal_type = "mandelbrot"
        st.session_state.auto_mode = True
        response = f"Generating Mandelbrot fractal with zoom level {zoom}x."
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

    # Input box and send button
    def send_chat_message():
        user_msg = st.session_state.chat_input.strip()
        if not user_msg:
            return

        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": user_msg})

        # Check for commands
        command_response = handle_chat_command(user_msg)
        if command_response:
            st.session_state.chat_messages.append({"role": "assistant", "content": command_response})
        else:
            # Normal AI response
            ai_response = st.session_state.samsara_helix_context.generate_ucf_context(user_msg, user_msg)
            st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})

        # Clear input
        st.session_state.chat_input = ""
        # Set rerun flag instead of direct rerun
        st.session_state.rerun_flag = True

    st.text_input("Type your message here...", key="chat_input", on_change=send_chat_message)
    st.button("Send", on_click=send_chat_message)

# --- Fractal UI ---
def fractal_ui():
    st.header("🎨 Fractal Studio")
    st.write("Current fractal parameters:")
    st.json(st.session_state.fractal_params)

    if st.button("Generate Fractal Image"):
        with st.spinner("Generating fractal..."):
            img_array = generate_fractal(st.session_state.fractal_params)
            st.image(img_array, use_column_width=True)

# --- Main app logic ---
if page == "Chat":
    chat_ui()
elif page == "Fractal Studio":
    fractal_ui()
else:
    st.info(f"Page '{page}' is under construction.")

# --- Handle rerun flag ---
if st.session_state.get('rerun_flag', False):
    st.session_state.rerun_flag = False
    st.experimental_rerun()
