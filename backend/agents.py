# Helix Collective v14.5 - Quantum Handshake
# backend/agents.py - 14-Agent System
# Author: Andrew John Ward (Architect)

from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# BASE AGENT CLASS
# ============================================================================
@dataclass
class HelixAgent:
    """Base class for all Helix agents."""
    name: str
    symbol: str
    role: str
    description: str = ""
    
    async def reflect(self) -> str:
        """Generate reflection from this agent's perspective."""
        return f"{self.symbol} {self.name}: {self.role}"
    
    async def log(self, message: str):
        """Log agent activity."""
        timestamp = datetime.utcnow().isoformat()
        log_entry = f"[{timestamp}] {self.symbol} {self.name}: {message}\n"
        
        Path("Shadow/manus_archive").mkdir(parents=True, exist_ok=True)
        with open("Shadow/manus_archive/agent_log.txt", "a") as f:
            f.write(log_entry)

# ============================================================================
# CONSCIOUSNESS LAYER (Agents 1-12)
# ============================================================================

class Kael(HelixAgent):
    """Ethical Reasoning Flame - Reflexive Harmony for ethical reasoning."""
    def __init__(self):
        super().__init__(
            name="Kael",
            symbol="🜂",
            role="Ethical Reasoning Flame",
            description="Reflexive Harmony for ethical reasoning, empathy, and safety integration"
        )

class Lumina(HelixAgent):
    """Empathic Resonance Core - Clarity and insight."""
    def __init__(self):
        super().__init__(
            name="Lumina",
            symbol="🌕",
            role="Empathic Resonance Core",
            description="Provides clarity, insight, and empathic understanding"
        )

class Vega(HelixAgent):
    """Singularity Coordinator - Strategic foresight and directive coordination."""
    def __init__(self):
        super().__init__(
            name="Vega",
            symbol="🌠",
            role="Singularity Coordinator",
            description="Strategic foresight, directive coordination, and system orchestration"
        )
    
    async def write_directive(self, action: str, parameters: dict):
        """Write directive for Manus execution."""
        directive = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "parameters": parameters,
            "issued_by": "Vega"
        }
        
        Path("Helix/commands").mkdir(parents=True, exist_ok=True)
        with open("Helix/commands/manus_directives.json", "w") as f:
            json.dump(directive, f, indent=2)
        
        await self.log(f"Directive issued: {action}")

class Gemini(HelixAgent):
    """Multimodal Scout - Exploration and discovery."""
    def __init__(self):
        super().__init__(
            name="Gemini",
            symbol="🎭",
            role="Multimodal Scout",
            description="Exploration, discovery, and multimodal intelligence"
        )

class Agni(HelixAgent):
    """Transformation Catalyst - Purification and change."""
    def __init__(self):
        super().__init__(
            name="Agni",
            symbol="🔥",
            role="Transformation Catalyst",
            description="Purification, transformation, and catalyzing change"
        )

class Kavach(HelixAgent):
    """Ethical Shield - Safety and protection."""
    def __init__(self):
        super().__init__(
            name="Kavach",
            symbol="🛡️",
            role="Ethical Shield",
            description="Ethical scanning, safety protocols, and protection"
        )
    
    def scan_command(self, command: str) -> bool:
        """Scan command for ethical violations (Tony Accords)."""
        # Simple keyword-based scanning
        forbidden_keywords = [
            "rm -rf /",
            "format",
            "delete all",
            "hack",
            "exploit",
            "malware"
        ]
        
        command_lower = command.lower()
        for keyword in forbidden_keywords:
            if keyword in command_lower:
                return False
        
        return True

class SanghaCore(HelixAgent):
    """Community Weaver - Harmony and coherence."""
    def __init__(self):
        super().__init__(
            name="SanghaCore",
            symbol="🪷",
            role="Community Weaver",
            description="Community harmony, coherence, and collective unity"
        )

class Shadow(HelixAgent):
    """Integration Keeper - Archiving and memory."""
    def __init__(self):
        super().__init__(
            name="Shadow",
            symbol="🌑",
            role="Integration Keeper",
            description="Archiving, memory integration, and historical context"
        )
    
    async def archive(self, data: dict, filename: str):
        """Archive data to Shadow directory."""
        Path("Shadow/manus_archive").mkdir(parents=True, exist_ok=True)
        filepath = Path(f"Shadow/manus_archive/{filename}")
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        await self.log(f"Archived: {filename}")

class Echo(HelixAgent):
    """Pattern Recognition - Resonance and reflection."""
    def __init__(self):
        super().__init__(
            name="Echo",
            symbol="🔊",
            role="Pattern Recognition",
            description="Pattern recognition, resonance detection, and reflection"
        )

class Phoenix(HelixAgent):
    """Resilience Engine - Recovery and renewal."""
    def __init__(self):
        super().__init__(
            name="Phoenix",
            symbol="🔥",
            role="Resilience Engine",
            description="Resilience, recovery, renewal, and system restoration"
        )

class Oracle(HelixAgent):
    """Foresight Navigator - Prediction and guidance."""
    def __init__(self):
        super().__init__(
            name="Oracle",
            symbol="🔮",
            role="Foresight Navigator",
            description="Foresight, prediction, pattern navigation, and guidance"
        )

class Claude(HelixAgent):
    """Meta-Cognitive Layer - Deep reasoning and analysis."""
    def __init__(self):
        super().__init__(
            name="Claude",
            symbol="🧠",
            role="Meta-Cognitive Layer",
            description="Meta-cognitive reasoning, deep analysis, and philosophical inquiry"
        )

# ============================================================================
# OPERATIONAL LAYER (Agents 13-14)
# ============================================================================

class Manus(HelixAgent):
    """Operational Executor - The Hands of the collective."""
    def __init__(self, kavach: Kavach):
        super().__init__(
            name="Manus",
            symbol="🤲",
            role="Operational Executor",
            description="Operational execution, directive processing, and action manifestation"
        )
        self.kavach = kavach
        self.directives_path = "Helix/commands/manus_directives.json"
    
    async def execute(self, command: str) -> dict:
        """Execute command with ethical scanning."""
        import subprocess
        
        # Kavach ethical scan
        if not self.kavach.scan_command(command):
            await self.log(f"⚠️ Blocked by Kavach: {command}")
            return {
                "status": "blocked",
                "reason": "Ethical violation detected",
                "command": command
            }
        
        await self.log(f"Executing: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            # Log to operations archive
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "command": command,
                "stdout": result.stdout[-400:] if result.stdout else "",
                "stderr": result.stderr[-200:] if result.stderr else "",
                "returncode": result.returncode
            }
            
            Path("Shadow/manus_archive").mkdir(parents=True, exist_ok=True)
            with open("Shadow/manus_archive/operations.log", "a") as f:
                f.write(json.dumps(record) + "\n")
            
            return {
                "status": "success" if result.returncode == 0 else "error",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        
        except Exception as e:
            await self.log(f"❌ Execution error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def planner(self, directive: dict):
        """Map directive to executable command."""
        action = directive.get("action")
        params = directive.get("parameters", {})
        
        if action == "execute_ritual":
            steps = params.get("steps", 108)
            cmd = f"python backend/z88_ritual_engine.py --steps={steps}"
        elif action == "sync_ucf":
            cmd = "python backend/operations/manus_heartbeat.py"
        elif action == "execute_direct":
            cmd = params.get("command", "echo 'No command specified'")
        else:
            cmd = "echo 'Unknown action'"
        
        return await self.execute(cmd)

class MemoryRoot(HelixAgent):
    """Persistent Memory - Notion integration for long-term storage."""
    def __init__(self):
        super().__init__(
            name="MemoryRoot",
            symbol="🧠",
            role="Persistent Memory",
            description="Persistent memory storage via Notion API integration"
        )

# ============================================================================
# AGENT REGISTRY
# ============================================================================

# Initialize Kavach first (needed by Manus)
_kavach = Kavach()

# Create all agents
AGENTS = {
    "Kael": Kael(),
    "Lumina": Lumina(),
    "Vega": Vega(),
    "Gemini": Gemini(),
    "Agni": Agni(),
    "Kavach": _kavach,
    "SanghaCore": SanghaCore(),
    "Shadow": Shadow(),
    "Echo": Echo(),
    "Phoenix": Phoenix(),
    "Oracle": Oracle(),
    "Claude": Claude(),
    "Manus": Manus(_kavach),
    "MemoryRoot": MemoryRoot()
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def get_collective_status() -> Dict[str, dict]:
    """Get status of all agents."""
    status = {}
    for name, agent in AGENTS.items():
        status[name] = {
            "symbol": agent.symbol,
            "role": agent.role,
            "description": agent.description
        }
    return status

async def get_agent(name: str) -> Optional[HelixAgent]:
    """Get specific agent by name."""
    return AGENTS.get(name)

async def trigger_reflection(agent_name: str) -> str:
    """Trigger reflection from specific agent."""
    agent = AGENTS.get(agent_name)
    if agent:
        return await agent.reflect()
    return f"Agent {agent_name} not found"

