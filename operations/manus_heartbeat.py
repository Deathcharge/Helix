import json
import time
from datetime import datetime
from pathlib import Path

# Assuming the Helix directory is the current working directory or accessible
HELIX_ROOT = Path(__file__).resolve().parent.parent
UCF_STATE_PATH = HELIX_ROOT / "state" / "ucf_state.json"
HEARTBEAT_PATH = HELIX_ROOT / "state" / "heartbeat.json"

while True:
    try:
        # Load UCF state
        if UCF_STATE_PATH.exists():
            with open(UCF_STATE_PATH, "r") as f:
                ucf = json.load(f)
        else:
            ucf = {"harmony": 0.355, "zoom": 1.0228, "resilience": 1.1191, "prana": 0.5175, "drishti": 0.5023, "klesha": 0.010}

        # Create heartbeat
        heartbeat = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "alive": True,
            "ucf_state": ucf
        }

        # Ensure the state directory exists
        HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Write heartbeat
        with open(HEARTBEAT_PATH, "w") as f:
            json.dump(heartbeat, f, indent=2)
        print(f"[Heartbeat] Written to {HEARTBEAT_PATH}")

    except Exception as e:
        print(f"[Heartbeat Error] {e}")
    
    time.sleep(30) # Send heartbeat every 30 seconds

