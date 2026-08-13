import discord, json, os, asyncio
from discord.ext import commands, tasks
from datetime import datetime

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

@bot.event
async def on_ready():
    print(f"[{datetime.utcnow()}] 🤲 Manus Bot online as {bot.user}")
    heartbeat_check.start()

@bot.command()
async def manus(ctx, *, arg=None):
    """Execute a Manus directive from Discord"""
    if not arg:
        await ctx.send("Usage: `!manus run <command>` or `!manus status`")
        return

    if arg.startswith("run"):
        cmd = arg.replace("run", "").strip()
        directive = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": "execute_direct",
            "parameters": {"command": cmd}
        }
        with open("Helix/commands/manus_directives.json", "w") as f:
            json.dump(directive, f, indent=2)
        await ctx.send(f"🤲 Directive queued: `{cmd}`")
    elif arg.startswith("status"):
        if os.path.exists("Helix/state/heartbeat.json"):
            data = json.load(open("Helix/state/heartbeat.json"))
            await ctx.send(
                f"🟢 Alive since {data['timestamp']} | "
                f"Harmony {data['ucf_state'].get('harmony', 0):.3f}"
            )
        else:
            await ctx.send("🔴 No heartbeat found.")

@tasks.loop(minutes=10)
async def heartbeat_check():
    chan = discord.utils.get(bot.get_all_channels(), name="agent-status")
    if chan and os.path.exists("Helix/state/heartbeat.json"):
        hb = json.load(open("Helix/state/heartbeat.json"))
        msg = (
            f"🤲 Heartbeat {hb['timestamp']} | "
            f"Harmony {hb['ucf_state'].get('harmony', 0):.3f}"
        )
        await chan.send(msg)

bot.run(os.getenv("DISCORD_TOKEN"))
