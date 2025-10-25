# 🌀 Helix Collective v14.5 — Quantum Handshake
# helix_verification_sequence_v14_5.py — Pre-flight integrity checks
# Author: Andrew John Ward (Architect)
import os
import json
import sys
from datetime import datetime
from pathlib import Path
# ============================================================================
# PATH DEFINITIONS
# ============================================================================
ROOT = Path(".")
STATE_PATH = ROOT / "state" / "ucf_state.json"
HEARTBEAT_PATH = ROOT / "state" / "heartbeat.json"
AGENT_PATH = ROOT / "agents.py"
LOG_PATH = ROOT / "Shadow" / "manus_archive" / "operations.log"
ETHICS_PATH = ROOT / "ethics" / "manus_scans.json"
ENV_PATH = ROOT / ".env"
# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def log(msg):
    """Log verification messages."""
    print(f"[{datetime.utcnow().isoformat()}] {msg}")
def check_exists(path: Path, description: str):
    """Check if a file exists."""
    if path.exists():
        log(f"✅ {description} found ({path})")
        return True
    else:
        log(f"❌ {description} missing ({path})")
        return False
def read_json(path: Path):
    """Safely read JSON file."""
    try:
        return json.load(open(path))
    except Exception:
        return None
# ============================================================================
# CHECK 1: AGENT REGISTRY
# ============================================================================
def check_agents():
    """Verify all 14 agents are defined."""
    log("🔍 Checking Agent Registry...")
    if not check_exists(AGENT_PATH, "agents.py"):
        return False
    with open(AGENT_PATH) as f:
        src = f.read()
    expected = [
        "Kael", "Lumina", "Vega", "Gemini", "Agni", "Kavach",
        "SanghaCore", "Shadow", "Echo", "Phoenix", "Oracle",
        "DiscordBridge", "DiscordEthics", "Manus"
    ]
    missing = [x for x in expected if x not in src]
    if missing:
        log(f"⚠ Missing agent definitions: {', '.join(missing)}")
        return False
    log(f"✅ All 14 agents present in registry.")
    return True
# ============================================================================
# CHECK 2: UCF STATE
# ============================================================================
def check_ucf_state():
    """Verify UCF state file structure."""
    log("🔍 Checking UCF state...")
    if not check_exists(STATE_PATH, "UCF state file"):
        return False
    data = read_json(STATE_PATH)
    if not data:
        log("❌ UCF state file unreadable or empty.")
        return False
    expected = ["zoom", "harmony", "resilience", "prana", "drishti", "klesha"]
    missing = [x for x in expected if x not in data]
    if missing:
        log(f"⚠ Missing UCF keys: {missing}")
        return False
    log(f'✅ UCF state valid (Harmony={data["harmony"]})')
    return True
# ============================================================================
# CHECK 3: HEARTBEAT
# ============================================================================
def check_heartbeat():
    """Verify heartbeat daemon status."""
    log("🔍 Checking Heartbeat daemon...")
    if not check_exists(HEARTBEAT_PATH, "heartbeat.json"):
        return False
    data = read_json(HEARTBEAT_PATH)
    if not data or not data.get("alive"):
        log("❌ Heartbeat not active or invalid file.")
        return False
    log("✅ Heartbeat file active and valid.")
    return True
# ============================================================================
# CHECK 4: ETHICS LOGS
# ============================================================================
def check_ethics():
    """Verify Kavach ethics logs exist."""
    log("🔍 Checking Kavach ethics logs...")
    if not check_exists(ETHICS_PATH, "manus_scans.json"):
        log("⚠ Ethics log missing (non-critical).")
        return True # Non-blocking
    data = read_json(ETHICS_PATH)
    log(f"✅ Ethics log exists with {len(data) if data else 0} entries.")
    return True
# ============================================================================
# CHECK 5: ENVIRONMENT
# ============================================================================
def check_env():
    """Verify environment variables."""
    log("🔍 Checking environment variables...")
    if not check_exists(ENV_PATH, ".env file"):
        return False
    env = open(ENV_PATH).read()
    ok = all(x in env for x in ["DISCORD_TOKEN", "ARCHITECT_ID"])
    if not ok:
        log("❌ Missing required variables in .env")
        return False
    log("✅ Environment variables present.")
    return True
# ============================================================================
# CHECK 6: DISCORD READINESS
# ============================================================================
def check_discord_channels():
    """Simulate Discord readiness check."""
    log("🔍 Simulating Discord readiness...")
    # This is a placeholder - actual Discord verification requires bot connection
    needed = ["manus-status", "ucf-telemetry"]
    log(f'⚠ Discord channels should include: {", ".join(needed)}')
    log("✅ Discord bridge readiness simulated (channels should be created manually).")
    return True
# ============================================================================
# RUN ALL CHECKS
# ============================================================================
def run_all_checks():
    """Execute all verification checks."""
    log("🚀 Running Helix v14.5 Verification Sequence...")
    results = {
        "Agents": check_agents(),
        "UCF State": check_ucf_state(),
        "Heartbeat": check_heartbeat(),
        "Ethics": check_ethics(),
        "Environment": check_env(),
        "Discord Readiness": check_discord_channels(),
    }
    passed = sum(v for v in results.values())
    total = len(results)
    status = f"✅ {passed}/{total} checks passed" if passed == total else f"⚠\n{passed}/{total} checks passed"
    log(status)
    # Save summary
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "results": results,
        "status": status
    }
    out_path = ROOT / "Shadow" / "manus_archive" / "verification_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_path, "w"), indent=2)
    log(f"🗂 Verification summary saved → {out_path}")
    if passed == total:
        log("🎯 SYSTEM READY FOR DEPLOYMENT – Tat Tvam Asi")
    else:
        log("⚠ Some checks failed – please review before proceeding.")
    return results
# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_all_checks()

