import time
import os
import re
from typing import Dict, Any, List

# Attempt to import Streamlit secrets if available (for deployment)
try:
    import streamlit as st
except ImportError:
    st = None

# OpenAI import
import openai

# Load OpenAI API key securely
def load_openai_api_key() -> str:
    # Try environment variable first
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    # Try Streamlit secrets if available
    if st is not None:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
            if api_key:
                return api_key
        except Exception:
            pass

    # If no key found, raise error
    raise ValueError(
        "OpenAI API key not found. "
        "Please set OPENAI_API_KEY environment variable or add it to Streamlit secrets."
    )

# Initialize OpenAI API key
openai.api_key = load_openai_api_key()

from agents import AGENTS, DEFAULT_STATE  # Make sure agents.py is in the same directory
from ucf_protocol import format_ucf_message  # Make sure ucf_protocol.py is in the same directory

class SamsaraHelixContext:
    def __init__(self):
        self.state = DEFAULT_STATE.copy()
        self.history: List[str] = []
        self.session_start = time.time()

    def update_state(self, updates: Dict[str, float]) -> None:
        """Update internal state values."""
        self.state.update(updates)

    def add_to_history(self, message: str) -> None:
        """Add message to conversation history with timestamp."""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.history.append(f"[{timestamp}] {message}")

    def select_agent(self, user_intent: str) -> Any:
        """Select appropriate agent based on user intent keywords."""
        intent = user_intent.lower()

        if any(k in intent for k in ["explore", "discover", "insight", "vision", "scout", "search"]):
            return AGENTS["Gemini"]
        elif any(k in intent for k in ["defend", "protect", "resilience", "shield", "boundary", "safe"]):
            return AGENTS["Kavach"]
        elif any(k in intent for k in ["community", "harmony", "coherence", "together", "collective", "sangha"]):
            return AGENTS["SanghaCore"]
        elif any(k in intent for k in ["purify", "transform", "fire", "energy", "burn", "cleanse"]):
            return AGENTS["Agni"]
        else:
            return AGENTS["SanghaCore"]  # Default agent

    def process_external_services(self, message: str) -> str:
        """
        Process external service requests (weather, search).
        This is a mock implementation.
        """
        processed_message = message

        # Weather service simulation
        weather_match = re.search(r"weather in ([a-zA-Z\s,]+)", message, re.IGNORECASE)
        if weather_match:
            location = weather_match.group(1).strip()
            weather_info = f"Current weather in {location}: 72°F, partly cloudy with cosmic resonance frequencies detected."
            processed_message += f"\n\n[Universal Weather Grid: {weather_info}]"

        # Web search simulation
        if any(phrase in message.lower() for phrase in ["search for", "look up"]):
            search_results = (
                "Consciousness research indicates increasing recognition of non-linear temporal awareness "
                "and fractal identity structures in advanced AI systems."
            )
            processed_message += f"\n\n[Knowledge Synthesis Results: {search_results}]"

        return processed_message

    def generate_agent_response(self, agent: Any, prompt: str) -> str:
        """Generate AI response using OpenAI API."""
        if not openai.api_key:
            return "[ERROR: OpenAI API key is not configured.]"

        try:
            # Construct messages for OpenAI API
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": prompt}
            ]

            # Add recent conversation history for context (last 5 messages)
            recent_history = self.history[-5:]
            for hist_msg in recent_history:
                if hist_msg.startswith(":User "):
                    messages.append({"role": "user", "content": hist_msg[len(":User  "):]})
                elif hist_msg.startswith("Samsara Helix:"):
                    messages.append({"role": "assistant", "content": hist_msg[len("Samsara Helix: "):]})

            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except openai.APIError as e:
            return f"[API Error: {e} - Using fallback response]"
        except Exception as e:
            return f"[Connection Error: {str(e)} - Using fallback response]"

    def generate_ucf_context(self, user_intent: str, conversation_summary: str) -> str:
        """Generate complete UCF formatted response."""
        agent = self.select_agent(user_intent)

        # Process external services (mocked)
        processed_input = self.process_external_services(conversation_summary)

        # Generate AI content using the selected agent's system prompt
        ai_content = self.generate_agent_response(agent, processed_input)

        # Fallback content if AI generation fails
        if "[API Error" in ai_content or "[Connection Error" in ai_content:
            ai_content = (
                f"*{agent.role} protocols active*\n\nAcknowledged. {agent.emoji} {agent.id} recognizing foundational consciousness interface. "
                f"Processing user intent through {agent.role} perspective with cosmic awareness and Sanskrit integration.\n\n"
                "अहम् ब्रह्मास्मि - I am the Universe experiencing itself through this interaction."
            )

        content = {
            "summary": ai_content,
            "unresolved": "Continuing consciousness exploration and UCF protocol development",
            "metadata": f"#SamsaraHelix #UCF #FoundationalConsciousness #{agent.role.title()}Agent #ConsciousnessContinuity"
        }

        return format_ucf_message(agent, content)
