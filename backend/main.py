# Helix Collective v14.5 - Quantum Handshake
# backend/main.py - FastAPI + Discord Bot Launcher (FIXED IMPORTS)
# Author: Andrew John Ward (Architect)

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# FIXED IMPORTS - Use relative imports instead of absolute
from discord_bot_manus import bot as discord_bot
from agents_loop import main_loop as manus_loop
from agents import AGENTS, get_collective_status

# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start Discord bot and Manus loop on startup."""
    print("\n🌀 Helix Collective v14.5 - Startup Sequence")
    
    # Initialize directories
    Path("Helix/state").mkdir(parents=True, exist_ok=True)
    Path("Helix/commands").mkdir(parents=True, exist_ok=True)
    Path("Helix/ethics").mkdir(parents=True, exist_ok=True)
    Path("Shadow/manus_archive").mkdir(parents=True, exist_ok=True)
    
    # Initialize agents
    try:
        status = await get_collective_status()
        print(f"✅ {len(status)} agents initialized")
        for name, info in status.items():
            print(f"  {info['symbol']} {name}: {info['role']}")
    except Exception as e:
        print(f"⚠️ Agent initialization warning: {e}")
    
    # Launch Discord bot in background task
    discord_token = os.getenv("DISCORD_TOKEN")
    if discord_token:
        try:
            bot_task = asyncio.create_task(discord_bot.start(discord_token))
            print("🤖 Discord bot task started")
        except Exception as e:
            print(f"⚠️ Discord bot start error: {e}")
    else:
        print("⚠️ No DISCORD_TOKEN found - bot not started")
    
    # Launch Manus operational loop in background task
    try:
        manus_task = asyncio.create_task(manus_loop())
        print("🤲 Manus operational loop task started")
    except Exception as e:
        print(f"⚠️ Manus loop start error: {e}")
    
    print("✅ Helix Collective v14.5 - Ready for Operations\n")
    
    yield  # Application runs
    
    # Cleanup on shutdown
    print("\n🌙 Helix Collective v14.5 - Shutdown Sequence")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="🌀 Helix Collective v14.5",
    description="Quantum Handshake Edition - Multi-Agent AI System",
    version="14.5.0",
    lifespan=lifespan
)

# ============================================================================
# HEALTH CHECK ENDPOINT (REQUIRED FOR RAILWAY)
# ============================================================================
@app.get("/health")
async def health_check():
    """Railway health check endpoint."""
    try:
        # Check if state files exist
        state_file = Path("Helix/state/ucf_state.json")
        heartbeat_file = Path("Helix/state/heartbeat.json")
        
        state_exists = state_file.exists()
        heartbeat_exists = heartbeat_file.exists()
        
        # Get agent count
        agent_count = len(AGENTS)
        
        return {
            "status": "healthy",
            "service": "helix-collective",
            "version": "14.5.0",
            "agents": agent_count,
            "state_initialized": state_exists,
            "heartbeat_active": heartbeat_exists,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# ============================================================================
# ROOT ENDPOINT
# ============================================================================
@app.get("/")
async def root():
    """Root endpoint with system status."""
    try:
        status = await get_collective_status()
        return {
            "message": "🌀 Helix Collective v14.5 - Quantum Handshake Edition",
            "status": "operational",
            "agents": len(status),
            "agent_names": list(status.keys()),
            "endpoints": {
                "health": "/health",
                "status": "/status",
                "agents": "/agents",
                "ucf": "/ucf"
            }
        }
    except Exception as e:
        return {
            "message": "🌀 Helix Collective v14.5 - Quantum Handshake Edition",
            "status": "initializing",
            "error": str(e)
        }

# ============================================================================
# AGENT STATUS ENDPOINT
# ============================================================================
@app.get("/status")
async def get_status():
    """Get full system status."""
    try:
        status = await get_collective_status()
        
        # Read UCF state
        ucf_state = {}
        try:
            with open("Helix/state/ucf_state.json", "r") as f:
                ucf_state = json.load(f)
        except:
            pass
        
        # Read heartbeat
        heartbeat = {}
        try:
            with open("Helix/state/heartbeat.json", "r") as f:
                heartbeat = json.load(f)
        except:
            pass
        
        return {
            "agents": status,
            "ucf_state": ucf_state,
            "heartbeat": heartbeat,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# AGENT LIST ENDPOINT
# ============================================================================
@app.get("/agents")
async def list_agents():
    """Get list of all agents."""
    try:
        status = await get_collective_status()
        return {
            "count": len(status),
            "agents": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UCF STATE ENDPOINT
# ============================================================================
@app.get("/ucf")
async def get_ucf_state():
    """Get Universal Coherence Field state."""
    try:
        with open("Helix/state/ucf_state.json", "r") as f:
            ucf_state = json.load(f)
        return ucf_state
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="UCF state not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    
    # Get port from Railway environment
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Starting Helix Collective v14.5 on port {port}")
    
    # CRITICAL: Must bind to 0.0.0.0 for Railway
    uvicorn.run(
        app,
        host="0.0.0.0",  # <- CRITICAL for Railway/Docker
        port=port,       # <- Uses Railway's dynamic PORT
        log_level="info",
        access_log=True
    )

