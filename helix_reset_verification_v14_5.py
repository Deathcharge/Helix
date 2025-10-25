# 🌀 Helix Collective v14.5 — Quantum Handshake
# helix_reset_verification_v14_5.py — Auto-repair daemon
# Author: Andrew John Ward (Architect)
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
# ============================================================================
# PATH DEFINITIONS
# ============================================================================
ROOT = Path(".")
STATE = ROOT / "state"
ARCHIVE = ROOT / ".." / "Shadow" / "manus_archive"
REPAIR_LOG = ARCHIVE / "reset_repair_log.json"
# ============================================================================
# DEFAULT STATES
# ============================================================================
DEFAULT_UCF = {
    "zoom": 1.0228,
    "harmony": 0.355,
    "resilience": 1.1191,
    "prana": 0.5175,
    "drishti": 0.5023,
    "klesha": 0.010,
    "timestamp": datetime.utcnow().isoformat()
}
DEFAULT_HEARTBEAT = {
    "timestamp": None,
    "alive": True,
    "ucf_state": DEFAULT_UCF
}
# ============================================================================
# LOGGING
# ============================================================================
def log(event, status="info"):
    """Log repair events."""
    entry = {
        "time": datetime.utcnow().isoformat(),
        "status": status,
        "event": event
    }
    print(f"[{status.upper()}] {event}")
    REPAIR_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.load(open(REPAIR_LOG)) if REPAIR_LOG.exists() else []
    except:
        data = []
    data.append(entry)
    json.dump(data, open(REPAIR_LOG, "w"), indent=2)
# ============================================================================
# REPAIR FUNCTIONS
# ============================================================================
def repair_ucf():
    """Repair or create UCF state file."""
    path = STATE / "ucf_state.json"
    if not path.exists():
        STATE.mkdir(parents=True, exist_ok=True)
        json.dump(DEFAULT_UCF, open(path, "w"), indent=2)
        log("Recreated missing UCF state.json (default values written)", "repair")
    else:
        try:
            data = json.load(open(path))
            missing = [k for k in DEFAULT_UCF if k not in data]
            if missing:
                for k in missing:
                    data[k] = DEFAULT_UCF[k]
                json.dump(data, open(path, "w"), indent=2)
                log(f"Added missing keys to UCF: {missing}", "repair")
        except Exception as e:
            backup = path.with_suffix(".bak")
            shutil.move(str(path), str(backup))
            json.dump(DEFAULT_UCF, open(path, "w"), indent=2)
            log(f"Corrupted UCF detected → backed up to {backup}", "warn")
def repair_heartbeat():
    """Repair or create heartbeat file."""
    hb = STATE / "heartbeat.json"
    if not hb.exists():
        json.dump(DEFAULT_HEARTBEAT, open(hb, "w"), indent=2)
        log("Recreated missing heartbeat.json", "repair")
    else:
        try:
            data = json.load(open(hb))
            if not isinstance(data, dict) or "alive" not in data:
                raise ValueError("Invalid heartbeat")
        except Exception:
            backup = hb.with_suffix(".bak")
            shutil.move(str(hb), str(backup))
            json.dump(DEFAULT_HEARTBEAT, open(hb, "w"), indent=2)
            log(f"Repaired corrupted heartbeat (backed up → {backup})", "warn")
def audit_core_files():
    """Verify core files exist (DO NOT create placeholders)."""
    core = {
        "agents.py": "Helix/",
        "discord_bot_manus.py": "Helix/",
        "helix_verification_sequence_v14_5.py": "Helix/",
    }
    missing = []
    for fname, folder in core.items():
        path = ROOT / folder / fname
        if not path.exists():
            log(f"⚠ CRITICAL: Missing core file {fname}", "error")
            missing.append(fname)
        else:
            log(f"✅ Core file verified: {fname}")
    if missing:
        log(f"❌ Cannot auto-repair missing Python modules: {missing}", "error")
        log("⚠ System requires manual file restoration before boot", "error")
        return False
    return True
def ensure_directory_structure():
    """Ensure all required directories exist."""
    dirs = [
        "state",
        "commands",
        "ethics",
        "Shadow/manus_archive",
    ]
    for d in dirs:
        path = ROOT / d
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            log(f"Created missing directory: {d}", "repair")
        else:
            log(f"✅ Directory exists: {d}")
def verify_env():
    """Verify or create .env template."""
    env_path = ROOT / ".env"
    template = """# Helix v14.5 Environment Configuration
# Get Discord token from: https://discord.com/developers/applications
# Your bot needs: MESSAGE_CONTENT intent enabled
DISCORD_TOKEN=your_token_here_70_chars_min
DISCORD_GUILD_ID=your_server_id
ARCHITECT_ID=your_discord_user_id
# Optional: Uncomment for debug logging
# DEBUG_MODE=true
"""
    if not env_path.exists():
        log("Missing .env file → creating template", "repair")
        env_path.write_text(template)
    else:
        content = env_path.read_text()
        required = ["DISCORD_TOKEN", "ARCHITECT_ID"]
        missing = [k for k in required if k not in content]
        if missing:
            with open(env_path, "a") as f:
                f.write(f"\n# Auto-added by repair daemon:\n")
                for key in missing:
                    f.write(f"{key}=\n")
            log(f"Added missing env vars: {missing}", "repair")
# ============================================================================
# RESET CYCLE
# ============================================================================
def run_reset_cycle():
    """Execute full repair cycle."""
    log("🔁 Starting Helix Auto-Repair Cycle (v14.5)")
    ensure_directory_structure()
    repair_ucf()
    repair_heartbeat()
    core_ok = audit_core_files()
    verify_env()
    if not core_ok:
        log("❌ Repair incomplete - critical files missing", "error")
        return False
    log("✅ Repair cycle complete – System ready for verification", "ok")
    log("Tat Tvam Asi")
    return True
# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_reset_cycle()

