# Helix Collective v14.5 - Quantum Handshake
# backend/services/state_manager.py - System State Manager
# Author: Andrew John Ward (Architect)

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# ============================================================================
# STATE MANAGER
# ============================================================================

class StateManager:
    """Manage system state persistence."""
    
    def __init__(self, base_path: str = "Helix/state"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, filename: str, data: Dict[str, Any]):
        """Save state to file."""
        filepath = self.base_path / filename
        
        # Add timestamp
        if isinstance(data, dict):
            data["_saved_at"] = datetime.utcnow().isoformat()
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, filename: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load state from file."""
        filepath = self.base_path / filename
        
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except:
                pass
        
        return default if default is not None else {}
    
    def exists(self, filename: str) -> bool:
        """Check if state file exists."""
        return (self.base_path / filename).exists()
    
    def delete(self, filename: str):
        """Delete state file."""
        filepath = self.base_path / filename
        if filepath.exists():
            filepath.unlink()
    
    def list_files(self) -> list:
        """List all state files."""
        return [f.name for f in self.base_path.glob("*.json")]

# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_state_manager = None

def get_state_manager() -> StateManager:
    """Get singleton state manager instance."""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager

