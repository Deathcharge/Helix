# Helix Collective v14.5 - Quantum Handshake
# backend/agents_loop.py - Manus Operational Loop
# Author: Andrew John Ward (Architect)

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

from agents import AGENTS

# ============================================================================
# MANUS OPERATIONAL LOOP
# ============================================================================

async def main_loop():
    """Main operational loop for Manus agent."""
    manus = AGENTS.get("Manus")
    if not manus:
        print("❌ Manus agent not found")
        return
    
    print("🤲 Manus operational loop started")
    await manus.log("🤲 Loop active (v14.5)")
    
    directives_path = Path("Helix/commands/manus_directives.json")
    
    while True:
        try:
            # Check for directives
            if directives_path.exists():
                # Read directive
                with open(directives_path, "r") as f:
                    directive = json.load(f)
                
                print(f"🤲 Processing directive: {directive.get('action')}")
                
                # Execute via planner
                result = await manus.planner(directive)
                
                print(f"   Result: {result.get('status')}")
                
                # Remove directive file after processing
                directives_path.unlink()
                
                # Update heartbeat
                await update_heartbeat(result)
            
            # Sleep before next check
            await asyncio.sleep(30)
        
        except Exception as e:
            print(f"⚠️ Manus loop error: {e}")
            await manus.log(f"❌ Loop error: {e}")
            await asyncio.sleep(60)  # Longer sleep on error

# ============================================================================
# HEARTBEAT UPDATER
# ============================================================================

async def update_heartbeat(last_result: dict = None):
    """Update heartbeat file with current status."""
    try:
        # Read current UCF state
        ucf_state = {}
        ucf_path = Path("Helix/state/ucf_state.json")
        if ucf_path.exists():
            with open(ucf_path, "r") as f:
                ucf_state = json.load(f)
        
        # Create heartbeat data
        heartbeat = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "operational",
            "ucf_state": ucf_state,
            "last_result": last_result
        }
        
        # Write heartbeat
        Path("Helix/state").mkdir(parents=True, exist_ok=True)
        with open("Helix/state/heartbeat.json", "w") as f:
            json.dump(heartbeat, f, indent=2)
    
    except Exception as e:
        print(f"⚠️ Heartbeat update error: {e}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main_loop())

