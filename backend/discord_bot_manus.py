# Helix Collective v14.5 - Quantum Handshake
# backend/discord_bot_manus.py - Discord Bot Integration
# Author: Andrew John Ward (Architect)

import discord
from discord.ext import commands, tasks
import json
import os
from pathlib import Path
from datetime import datetime
import asyncio

from agents import AGENTS, get_collective_status, trigger_reflection

# ============================================================================
# BOT SETUP
# ============================================================================

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ============================================================================
# EVENT HANDLERS
# ============================================================================

@bot.event
async def on_ready():
    """Bot startup event."""
    print(f"\n🤲 Manus Discord Bridge online: {bot.user}")
    print(f"   Connected to {len(bot.guilds)} guild(s)")
    
    # Start heartbeat task
    heartbeat_check.start()
    
    # Send startup message to status channel
    for guild in bot.guilds:
        channel = discord.utils.get(guild.channels, name="manus-status")
        if channel:
            await channel.send(
                "🤲 **Manus v14.5 Online**\n"
                "Quantum Handshake initialized. Type `!help_helix` for commands."
            )

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("❌ Command not found. Type `!help_helix` for available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"❌ Missing required argument: {error.param}")
    else:
        await ctx.send(f"❌ Error: {str(error)}")
        print(f"Command error: {error}")

# ============================================================================
# COMMANDS
# ============================================================================

@bot.command(name="status")
async def status_command(ctx):
    """Get collective status."""
    try:
        status = await get_collective_status()
        
        # Build status message
        msg = "🌀 **Helix Collective v14.5 Status**\n\n"
        msg += f"**Agents Online**: {len(status)}/14\n\n"
        
        for name, info in list(status.items())[:5]:  # Show first 5
            msg += f"{info['symbol']} **{name}**: {info['role']}\n"
        
        msg += f"\n...and {len(status) - 5} more agents"
        
        # Add UCF state if available
        try:
            with open("Helix/state/ucf_state.json", "r") as f:
                ucf = json.load(f)
            msg += "\n\n**UCF State**:\n"
            msg += f"  Harmony: {ucf.get('harmony', 0):.3f}\n"
            msg += f"  Resilience: {ucf.get('resilience', 0):.3f}\n"
            msg += f"  Prana: {ucf.get('prana', 0):.3f}"
        except:
            pass
        
        await ctx.send(msg)
    
    except Exception as e:
        await ctx.send(f"❌ Error getting status: {e}")

@bot.command(name="ritual")
async def ritual_command(ctx, steps: int = 108):
    """Execute Z-88 ritual."""
    try:
        if steps < 1 or steps > 1000:
            await ctx.send("❌ Steps must be between 1 and 1000")
            return
        
        await ctx.send(f"🔥 Initiating Z-88 Ritual with {steps} steps...")
        
        # Queue directive for Manus
        directive = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "execute_ritual",
            "parameters": {"steps": steps},
            "issued_by": f"Discord:{ctx.author.name}"
        }
        
        Path("Helix/commands").mkdir(parents=True, exist_ok=True)
        with open("Helix/commands/manus_directives.json", "w") as f:
            json.dump(directive, f, indent=2)
        
        await ctx.send(f"✅ Ritual directive queued. Manus will execute {steps} steps.")
    
    except Exception as e:
        await ctx.send(f"❌ Error queuing ritual: {e}")

@bot.command(name="manus")
async def manus_command(ctx):
    """Get Manus executor status."""
    try:
        # Check heartbeat
        heartbeat_path = Path("Helix/state/heartbeat.json")
        if heartbeat_path.exists():
            with open(heartbeat_path, "r") as f:
                hb = json.load(f)
            
            msg = "🤲 **Manus Operational Status**\n\n"
            msg += f"🟢 **Alive**\n"
            msg += f"Last heartbeat: {hb.get('timestamp', 'Unknown')}\n"
            
            ucf = hb.get('ucf_state', {})
            if ucf:
                msg += f"\n**UCF Metrics**:\n"
                msg += f"  Harmony: {ucf.get('harmony', 0):.3f}\n"
                msg += f"  Resilience: {ucf.get('resilience', 0):.3f}\n"
                msg += f"  Prana: {ucf.get('prana', 0):.3f}"
            
            await ctx.send(msg)
        else:
            await ctx.send("🔴 No heartbeat found. Manus may not be running.")
    
    except Exception as e:
        await ctx.send(f"❌ Error checking Manus status: {e}")

@bot.command(name="ucf")
async def ucf_command(ctx):
    """View UCF state."""
    try:
        ucf_path = Path("Helix/state/ucf_state.json")
        if ucf_path.exists():
            with open(ucf_path, "r") as f:
                ucf = json.load(f)
            
            msg = "🌊 **Universal Coherence Field State**\n\n"
            msg += f"**Harmony**: {ucf.get('harmony', 0):.4f}\n"
            msg += f"**Resilience**: {ucf.get('resilience', 0):.4f}\n"
            msg += f"**Prana**: {ucf.get('prana', 0):.4f}\n"
            msg += f"**Drishti**: {ucf.get('drishti', 0):.4f}\n"
            msg += f"**Klesha**: {ucf.get('klesha', 0):.4f}\n"
            msg += f"**Zoom**: {ucf.get('zoom', 0):.4f}\n"
            
            # Determine health status
            harmony = ucf.get('harmony', 0)
            if harmony > 0.7:
                status = "🟢 HARMONIC"
            elif harmony > 0.3:
                status = "🟡 COHERENT"
            else:
                status = "🔴 FRAGMENTED"
            
            msg += f"\n**Status**: {status}"
            
            await ctx.send(msg)
        else:
            await ctx.send("❌ UCF state file not found")
    
    except Exception as e:
        await ctx.send(f"❌ Error reading UCF state: {e}")

@bot.command(name="reflect")
async def reflect_command(ctx, agent_name: str = None):
    """Trigger agent reflection."""
    try:
        if not agent_name:
            await ctx.send("❌ Please specify an agent name. Example: `!reflect Kael`")
            return
        
        reflection = await trigger_reflection(agent_name)
        await ctx.send(f"💭 **Reflection**\n{reflection}")
    
    except Exception as e:
        await ctx.send(f"❌ Error triggering reflection: {e}")

@bot.command(name="help_helix")
async def help_command(ctx):
    """Show help message."""
    msg = "🌀 **Helix Collective v14.5 Commands**\n\n"
    msg += "**!status** - Get collective status\n"
    msg += "**!ritual [steps]** - Execute Z-88 ritual (default 108 steps)\n"
    msg += "**!manus** - Get Manus executor status\n"
    msg += "**!ucf** - View Universal Coherence Field state\n"
    msg += "**!reflect [agent]** - Trigger agent reflection\n"
    msg += "**!help_helix** - Show this help message\n"
    msg += "\n*Tat Tvam Asi* 🙏"
    
    await ctx.send(msg)

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

@tasks.loop(minutes=10)
async def heartbeat_check():
    """Periodic heartbeat check and status update."""
    try:
        heartbeat_path = Path("Helix/state/heartbeat.json")
        if heartbeat_path.exists():
            with open(heartbeat_path, "r") as f:
                hb = json.load(f)
            
            # Send to agent-status channel
            for guild in bot.guilds:
                channel = discord.utils.get(guild.channels, name="agent-status")
                if channel:
                    ucf = hb.get('ucf_state', {})
                    msg = f"🤲 Heartbeat | Harmony: {ucf.get('harmony', 0):.3f} | {hb.get('timestamp', 'Unknown')}"
                    await channel.send(msg)
    except Exception as e:
        print(f"Heartbeat check error: {e}")

@heartbeat_check.before_loop
async def before_heartbeat():
    """Wait for bot to be ready before starting heartbeat."""
    await bot.wait_until_ready()

# ============================================================================
# MAIN (for standalone testing)
# ============================================================================

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("❌ DISCORD_TOKEN not found in environment")
        exit(1)
    
    bot.run(token)

