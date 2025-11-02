# Helix Collective v14.5 - Quantum Handshake
# backend/services/ucf_calculator.py - Universal Coherence Field Calculator
# Author: Andrew John Ward (Architect)

import json
from pathlib import Path
from datetime import datetime
from typing import Dict

# ============================================================================
# UCF STATE DEFAULTS
# ============================================================================

DEFAULT_UCF_STATE = {
    "zoom": 1.0228,
    "harmony": 0.355,
    "resilience": 1.1191,
    "prana": 0.5175,
    "drishti": 0.5023,
    "klesha": 0.010,
    "last_updated": None
}

# ============================================================================
# UCF CALCULATOR
# ============================================================================

class UCFCalculator:
    """Calculate and manage Universal Coherence Field state."""
    
    def __init__(self, state_path: str = "Helix/state/ucf_state.json"):
        self.state_path = Path(state_path)
        self.state = self.load_state()
    
    def load_state(self) -> Dict[str, float]:
        """Load UCF state from file or create default."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    return json.load(f)
            except:
                pass
        
        # Create default state
        state = DEFAULT_UCF_STATE.copy()
        state["last_updated"] = datetime.utcnow().isoformat()
        self.save_state(state)
        return state
    
    def save_state(self, state: Dict[str, float] = None):
        """Save UCF state to file."""
        if state is None:
            state = self.state
        
        state["last_updated"] = datetime.utcnow().isoformat()
        
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)
    
    def update(self, **kwargs):
        """Update UCF state with new values."""
        for key, value in kwargs.items():
            if key in self.state:
                self.state[key] = value
        
        self.save_state()
    
    def calculate_health_status(self) -> str:
        """Calculate overall health status based on harmony."""
        harmony = self.state.get("harmony", 0)
        
        if harmony > 0.7:
            return "HARMONIC"
        elif harmony > 0.3:
            return "COHERENT"
        else:
            return "FRAGMENTED"
    
    def adjust_harmony(self, delta: float):
        """Adjust harmony value."""
        current = self.state.get("harmony", 0)
        new_value = max(0.0, min(1.0, current + delta))
        self.update(harmony=new_value)
    
    def adjust_resilience(self, delta: float):
        """Adjust resilience value."""
        current = self.state.get("resilience", 0)
        new_value = max(0.0, current + delta)
        self.update(resilience=new_value)
    
    def adjust_prana(self, delta: float):
        """Adjust prana (energy) value."""
        current = self.state.get("prana", 0)
        new_value = max(0.0, min(1.0, current + delta))
        self.update(prana=new_value)
    
    def adjust_klesha(self, delta: float):
        """Adjust klesha (entropy) value."""
        current = self.state.get("klesha", 0)
        new_value = max(0.0, current + delta)
        self.update(klesha=new_value)
    
    def get_state(self) -> Dict[str, float]:
        """Get current UCF state."""
        return self.state.copy()
    
    def reset_to_default(self):
        """Reset UCF state to default values."""
        self.state = DEFAULT_UCF_STATE.copy()
        self.save_state()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_ucf_calculator() -> UCFCalculator:
    """Get singleton UCF calculator instance."""
    return UCFCalculator()

def initialize_ucf_state():
    """Initialize UCF state file if it doesn't exist."""
    calculator = UCFCalculator()
    print(f"✅ UCF state initialized: {calculator.calculate_health_status()}")
    return calculator.get_state()

