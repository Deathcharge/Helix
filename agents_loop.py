# 🌀 Helix Collective v14.5 — Quantum Handshake
# agents_loop.py — Main Manus operational loop (FINAL PATCHED)
# Author: Andrew John Ward (Architect)
import asyncio
import json
import os
import datetime
from pathlib import Path
from agents import Manus, Kavach
# ============================================================================
# PATH DEFINITIONS
# ============================================================================
ARCHIVE_PATH = Path("Shadow/manus_archive/")
COMMANDS_PATH = Path("Helix/commands/manus_directives.json")
STATE_PATH = Path("Helix/state/ucf_state.json")
RITUAL_LOCK = Path("Helix/state/.ritual_lock")
# Ensure directories exist
for p in [ARCHIVE_PATH, COMMANDS_PATH.parent, STATE_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)
# ============================================================================
# HEARTBEAT HELPER
# ============================================================================
def update_heartbeat(status="active", harmony=0.355):
    """Update heartbeat.json with current status."""
    heartbeat_path = Path("Helix/state/heartbeat.json")
    data = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "alive": True,
        "status": status,
        "ucf_state": {"harmony": harmony}
    }
    json.dump(data, open(heartbeat_path, "w"), indent=2)
# ============================================================================
# LOGGING
# ============================================================================
async def log_event(message: str):
    """Log loop events with timestamp."""
    now = datetime.datetime.utcnow().isoformat()
    record = {"time": now, "event": message}
    log_file = ARCHIVE_PATH / "agents_loop.log"
    try:
        data = json.load(open(log_file)) if log_file.exists() else []
    except:
        data = []
    data.append(record)
    json.dump(data, open(log_file, "w"), indent=2)
    print(message)
# ============================================================================
# UCF STATE HELPERS
# ============================================================================
async def load_ucf_state():
    """Load UCF state, creating default if missing."""
    if not STATE_PATH.exists():
        base = {
            "zoom": 1.0228,
            "harmony": 0.355,
            "resilience": 1.1191,
            "prana": 0.5175,
            "drishti": 0.5023,
            "klesha": 0.010
        }
        json.dump(base, open(STATE_PATH, "w"), indent=2)
    return json.load(open(STATE_PATH))
async def save_ucf_state(state):
    """Save UCF state to disk."""
    json.dump(state, open(STATE_PATH, "w"), indent=2)
# ============================================================================
# DIRECTIVE PROCESSING
# ============================================================================
async def process_directives(manus, kavach):
    """Check for directives from Vega or Architect."""
    if not COMMANDS_PATH.exists():
        return

    try:
        directive = json.load(open(COMMANDS_PATH))
        # Kavach scan before execution
        if "command" in directive:
            command_to_scan = directive["command"]
            scan_result = await kavach.scan(command_to_scan)
            if not scan_result.get("approved"):
                await log_event(f"🛡 Kavach blocked: {command_to_scan}")
                os.remove(COMMANDS_PATH)
                return
        await manus.planner(directive)
        os.remove(COMMANDS_PATH)
        await log_event(f"✅ Processed directive: {directive}")
    except Exception as e:
        await log_event(f"⚠ Directive processing error: {e}")
# ============================================================================
# MAIN LOOP
# ============================================================================
async def main_loop():
    """Main Manus operational loop."""
    kavach = Kavach()
    manus = Manus(kavach)
    await log_event("🤲 Manus loop initiated (v14.5 patched)")
    while True:
        try:
            # Pause loop if ritual in progress
            if RITUAL_LOCK.exists():
                await log_event("⏸ Pausing loop — ritual in progress")
                await asyncio.sleep(5)
                continue
            # Process directives
            await process_directives(manus, kavach)
            # Update UCF state
            ucf = await load_ucf_state()
            # Conservative harmony growth (0.0001 per cycle)
            ucf["harmony"] = min(1.0, ucf["harmony"] + 0.0001)
            await save_ucf_state(ucf)
            # Update heartbeat
            update_heartbeat(status="active", harmony=ucf["harmony"])
        except Exception as e:
            await log_event(f"Error in Manus loop: {e}")
        await asyncio.sleep(30)
# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("🛑 Manus loop stopped manually.")

