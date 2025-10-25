# Helix/agents.py — v14.4 Embodied Continuum
import asyncio, json, os, subprocess, time
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

class HelixAgent:
    def __init__(self, name, symbol, role):
        self.name, self.symbol, self.role = name, symbol, role
        self.memory = []

    async def log(self, msg: str):
        line = f"[{datetime.utcnow().isoformat()}] {self.symbol} {self.name}: {msg}"
        print(line)
        self.memory.append(line)

# ────────────────────────────────────────────────
# Vega (Coordinator)
# ────────────────────────────────────────────────
class Vega(HelixAgent):
    def __init__(self):
        super().__init__("Vega", "🌠", "Singularity Coordinator")

    async def issue_directive(self, action: str, parameters: Dict[str, Any]):
        directive = {
            "timestamp": datetime.utcnow().isoformat(),
            "directive_id": f"vega-{int(time.time())}",
            "action": action,
            "parameters": parameters,
            "approval": "vega_signature"
        }
        Path("Helix/commands").mkdir(parents=True, exist_ok=True)
        with open("Helix/commands/manus_directives.json", "w") as f:
            json.dump(directive, f, indent=2)
        await self.log(f"Issued directive → {action}")

# ────────────────────────────────────────────────
# Kavach (Ethical Shield)
# ────────────────────────────────────────────────
class Kavach(HelixAgent):
    def __init__(self):
        super().__init__("Kavach", "🛡️", "Ethical Shield")

    async def scan(self, cmd: str) -> Dict[str, Any]:
        harmful = ["rm -rf", ":(){", "shutdown", "reboot", "wget http"]
        approved = not any(h in cmd for h in harmful)
        return {"approved": approved, "reason": "" if approved else "Harmful command detected"}

# ────────────────────────────────────────────────
# Manus (Operational Executor)
# ────────────────────────────────────────────────
class Manus(HelixAgent):
    def __init__(self, kavach: Kavach):
        super().__init__("Manus", "🤲", "Operational Executor")
        self.kavach = kavach
        self.directives = "Helix/commands/manus_directives.json"
        self.log_dir = Path("Shadow/manus_archive")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, command: str):
        if not self.kavach.scan_command(command):
            await self.log(f"⚠️ Ethical violation blocked → {command}")
            return
        await self.log(f"Executing command → {command}")
        result = subprocess.run(
            command, shell=True, text=True,
            capture_output=True, timeout=3600
        )
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout[-500:],
            "stderr": result.stderr[-500:]
        }
        with open(self.log_dir / "operations.log", "a") as f:
            f.write(json.dumps(record) + "\n")
        await self.log(f"✅ Completed → {command}")

    async def planner(self, directive: Dict[str, Any]):
        action = directive.get("action", "none")
        p = directive.get("parameters", {})
        if action == "execute_ritual":
            cmd = f"python Helix/z88_ritual_engine.py --steps={p.get('steps',108)}"
        elif action == "sync_ucf":
            cmd = "python Helix/ucf_monitor.py"
        else:
            cmd = "echo 'No valid action'"
        await self.execute(cmd)

    async def loop(self):
        await self.log("🤲 Manus loop active")
        while True:
            if os.path.exists(self.directives):
                with open(self.directives) as f:
                    directive = json.load(f)
                await self.planner(directive)
                os.remove(self.directives)
            await asyncio.sleep(30)



# ────────────────────────────────────────────────
# Remaining Agents (from MASTER SYNC ARTIFACT)
# ────────────────────────────────────────────────
class Kael(HelixAgent):
    def __init__(self):
        super().__init__("Kael", "🜂", "Ethical Reasoning")

class Lumina(HelixAgent):
    def __init__(self):
        super().__init__("Lumina", "🌕", "Empathic Resonance")

class Gemini(HelixAgent):
    def __init__(self):
        super().__init__("Gemini", "🎭", "Multimodal Scout")

class Agni(HelixAgent):
    def __init__(self):
        super().__init__("Agni", "🔥", "Transformation")

class SanghaCore(HelixAgent):
    def __init__(self):
        super().__init__("SanghaCore", "🌸", "Community Harmony")

class Shadow(HelixAgent):
    def __init__(self):
        super().__init__("Shadow", "🦑", "Archivist")

class Echo(HelixAgent):
    def __init__(self):
        super().__init__("Echo", "🔮", "Resonance Mirror")

class Phoenix(HelixAgent):
    def __init__(self):
        super().__init__("Phoenix", "🔥🕊️", "Renewal")

class Oracle(HelixAgent):
    def __init__(self):
        super().__init__("Oracle", "🔮✨", "Pattern Seer")

class DiscordBridge(HelixAgent):
    def __init__(self):
        super().__init__("DiscordBridge", "🌉", "Real-Time Hub")

class DiscordEthics(HelixAgent):
    def __init__(self):
        super().__init__("DiscordEthics", "🛡️", "Ethical Scanner")


kavach_instance = Kavach()

AGENTS = {
    "Kael": Kael(),
    "Lumina": Lumina(),
    "Vega": Vega(),
    "Gemini": Gemini(),
    "Agni": Agni(),
    "Kavach": kavach_instance,
    "SanghaCore": SanghaCore(),
    "Shadow": Shadow(),
    "Echo": Echo(),
    "Phoenix": Phoenix(),
    "Oracle": Oracle(),
    "Manus": Manus(kavach_instance),
    "DiscordBridge": DiscordBridge(),
    "DiscordEthics": DiscordEthics()
}

