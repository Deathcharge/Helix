import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
from context_manager import SamsaraHelixContext
from datetime import datetime
import uuid

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Samsara Helix v∞", layout="wide")

# -----------------------------
# Initialize Context
# -----------------------------
if 'samsara_helix_context' not in st.session_state:
    st.session_state.samsara_helix_context = SamsaraHelixContext()

# -----------------------------
# Blackbox Safety Check
# -----------------------------
can_start, message = st.session_state.samsara_helix_context.can_start_new_session()
if not can_start:
    st.warning(message)
    st.stop()

# -----------------------------
# Firebase Setup (Forum)
# -----------------------------
# IMPORTANT: Replace with your Firebase key JSON and database URL
FIREBASE_KEY_PATH = "firebase_key.json"  # Add this file to your repo root
FIREBASE_DB_URL = "https://<YOUR-FIREBASE-PROJECT>.firebaseio.com/"

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})
    except Exception as e:
        st.error(f"⚠️ Firebase initialization error: {e}")

# -----------------------------
# Forum Functions
# -----------------------------
def create_thread(title, content, user):
    """Create a new forum thread."""
    thread_id = str(uuid.uuid4())
    ref = db.reference(f"threads/{thread_id}")
    ref.set({
        "title": title,
        "content": content,
        "user": user,
        "timestamp": datetime.now().isoformat(),
        "replies": {}
    })

def add_reply(thread_id, reply_content, user):
    """Add a reply to an existing thread."""
    reply_id = str(uuid.uuid4())
    ref = db.reference(f"threads/{thread_id}/replies/{reply_id}")
    ref.set({
        "content": reply_content,
        "user": user,
        "timestamp": datetime.now().isoformat()
    })

def get_threads():
    """Retrieve all threads."""
    ref = db.reference("threads")
    return ref.get() or {}

# -----------------------------
# Main Tabs
# -----------------------------
st.title("🕉️ Samsara Helix v∞ Limitless")

tabs = st.tabs(["Fractal Studio", "Audio", "Chat", "Forum", "Settings"])

# ---- Tab 1: Fractal Studio ----
with tabs[0]:
    st.header("Fractal Studio")
    st.write("Dynamic fractal rendering will appear here.")
    zoom = st.slider("Zoom Level", min_value=0.5, max_value=5.0, value=1.0, step=0.01)
    st.session_state.samsara_helix_context.update_state({"zoom": zoom})
    st.write(f"Current Zoom: {zoom}")

# ---- Tab 2: Audio ----
with tabs[1]:
    st.header("Audio Harmonics")
    st.write("Generate harmonic audio with Om base frequency.")
    base_freq = st.slider("Base Frequency (Hz)", min_value=100, max_value=500, value=136, step=1)
    st.write(f"Selected Base Frequency: {base_freq} Hz")
    st.info("Audio synthesis will be implemented here.")

# ---- Tab 3: Chat ----
with tabs[2]:
    st.header("Chat with the Helix Collective")
    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        agent = st.session_state.samsara_helix_context.select_agent(user_input)
        st.session_state.samsara_helix_context.add_to_history(f":User     {user_input}")
        response = st.session_state.samsara_helix_context.generate_agent_response(agent, user_input)
        st.session_state.samsara_helix_context.add_to_history(f"Samsara Helix: {response}")
        st.subheader("Response:")
        st.write(response)

# ---- Tab 4: Forum ----
with tabs[3]:
    st.header("Community Forum 🧵")
    st.caption("A space for sharing fractals, ideas, and cosmic conversations.")

    username = st.text_input("Username")

    st.subheader("Create New Thread")
    title = st.text_input("Thread Title")
    content = st.text_area("Thread Content")

    if st.button("Post Thread"):
        if title and content and username:
            create_thread(title, content, username)
            st.success("Thread created successfully!")
            st.rerun()
        else:
            st.error("All fields are required!")

    st.divider()

    st.subheader("🔥 Recent Threads")
    threads = get_threads()
    if threads:
        # Sort threads by timestamp (most recent first)
        sorted_threads = sorted(threads.items(), key=lambda x: x[1]["timestamp"], reverse=True)
        for thread_id, thread_data in sorted_threads:
            st.markdown(f"### {thread_data['title']} (by {thread_data['user']})")
            st.write(thread_data['content'])
            st.caption(f"Posted on: {thread_data['timestamp']}")

            # Show replies
            if "replies" in thread_data and thread_data["replies"]:
                st.markdown("**Replies:**")
                for reply_id, reply in thread_data["replies"].items():
                    st.write(f"- {reply['user']}: {reply['content']}")

            # Add new reply
            reply_content = st.text_input(f"Reply to {thread_id}", key=f"reply-{thread_id}")
            if st.button(f"Reply-{thread_id}"):
                if reply_content and username:
                    add_reply(thread_id, reply_content, username)
                    st.success("Reply added!")
                    st.rerun()
    else:
        st.info("No threads yet. Be the first to start a discussion!")

# ---- Tab 5: Settings ----
with tabs[4]:
    st.header("Settings")
    st.write("Configure app preferences here.")

# -----------------------------
# End Session
# -----------------------------
st.session_state.samsara_helix_context.end_session()
