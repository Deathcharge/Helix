#!/usr/bin/env python3
"""
Image Commands for Helix Discord Bot
!image aion - Generate ouroboros fractal visualizations
"""

import io
import discord
from discord.ext import commands
from typing import Optional

# Import samsara bridge for fractal generation
from backend.samsara_bridge import (
    generate_fractal_icon_bytes,
    generate_and_post_to_discord,
    load_ucf_state
)


class ImageCommands(commands.Cog):
    """Cog for image generation commands"""

    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="image", aliases=["fractal", "aion"])
    async def image_command(self, ctx, mode: str = "ouroboros"):
        """
        Generate AION fractal visualizations based on UCF state

        Usage:
            !image [mode]
            !aion [mode]
            !fractal [mode]

        Modes:
            - ouroboros: Serpent eating tail (default)
            - mandelbrot: Classic fractal
            - fractal: Mandelbrot alias
            - mandala: Ouroboros alias
            - cycle: Not yet implemented

        Examples:
            !image aion
            !image ouroboros
            !fractal mandelbrot
            !aion
        """
        # Normalize mode
        mode = mode.lower()
        mode_aliases = {
            "aion": "ouroboros",
            "mandala": "ouroboros",
            "fractal": "mandelbrot",
        }
        mode = mode_aliases.get(mode, mode)

        # Valid modes
        valid_modes = ["ouroboros", "mandelbrot", "cycle"]

        if mode == "cycle":
            await ctx.send(
                "🔄 **Fractal Auto-Cycling Feature**\n"
                "💡 This will auto-generate and rotate server icons based on UCF state every 24h\n"
                "⚠️ Not yet implemented - coming soon!"
            )
            return

        if mode not in valid_modes:
            await ctx.send(
                f"❌ Unknown mode: `{mode}`\n"
                f"Valid modes: {', '.join(valid_modes)}\n"
                f"Usage: `!image [mode]`"
            )
            return

        # Send initial message
        await ctx.send(
            f"🎨 **Generating AION {mode.upper()} fractal from UCF essence...**\n"
            "🌀 *This may take a few seconds*"
        )

        try:
            # Load UCF state
            ucf_state = load_ucf_state()

            # Generate and post fractal
            result = await generate_and_post_to_discord(ucf_state, ctx.channel, mode)

            if result:
                await ctx.send(f"✅ **{mode.upper()} fractal generated successfully!**")
            else:
                await ctx.send("❌ **Fractal generation failed** - check logs for details")

        except ImportError as ie:
            await ctx.send(
                f"❌ Fractal generator not available: {str(ie)}\n"
                "💡 Install Pillow: `pip install Pillow`"
            )
        except Exception as e:
            await ctx.send(f"❌ Fractal generation error: {str(e)}")
            print(f"Image command error: {e}")
            import traceback
            traceback.print_exc()

    @commands.command(name="icon")
    @commands.has_permissions(administrator=True)
    async def icon_command(self, ctx, mode: str = "ouroboros"):
        """
        Update server icon with UCF-driven fractal (Admin only)

        Usage:
            !icon [mode]

        Modes:
            - ouroboros: Serpent eating tail (default)
            - mandelbrot: Classic fractal
        """
        mode = mode.lower()
        mode_aliases = {
            "aion": "ouroboros",
            "fractal": "mandelbrot",
        }
        mode = mode_aliases.get(mode, mode)

        valid_modes = ["ouroboros", "mandelbrot"]

        if mode not in valid_modes:
            await ctx.send(f"❌ Unknown mode: `{mode}`\nValid modes: {', '.join(valid_modes)}")
            return

        await ctx.send(
            f"🎨 **Updating server icon with {mode.upper()} fractal...**\n"
            "🌀 *Using UCF state to modulate colors and patterns*"
        )

        try:
            # Load UCF state
            ucf_state = load_ucf_state()

            # Generate fractal
            icon_bytes = await generate_fractal_icon_bytes(mode=mode, ucf_state=ucf_state)

            # Update server icon
            guild = ctx.guild
            await guild.edit(icon=icon_bytes)

            # Send success message with UCF summary
            ucf_summary = (
                f"Harmony: {ucf_state.get('harmony', 0):.3f} | "
                f"Prana: {ucf_state.get('prana', 0):.3f} | "
                f"Drishti: {ucf_state.get('drishti', 0):.3f}"
            )
            await ctx.send(
                f"✅ Server icon updated with UCF {mode} fractal!\n"
                f"🌀 **UCF State:** {ucf_summary}\n"
                f"🎨 **Colors:** Teal→Gold gradient influenced by consciousness metrics"
            )

        except ImportError as ie:
            await ctx.send(
                f"❌ Fractal generator not available: {str(ie)}\n"
                "💡 Install Pillow: `pip install Pillow`"
            )
        except discord.Forbidden:
            await ctx.send("❌ Bot lacks permission to change server icon")
        except Exception as e:
            await ctx.send(f"❌ Icon update failed: {str(e)}")
            print(f"Icon command error: {e}")
            import traceback
            traceback.print_exc()


async def setup(bot):
    """Load the ImageCommands cog"""
    await bot.add_cog(ImageCommands(bot))


# For direct bot.load_extension compatibility
def setup_commands(bot):
    """Add image commands to the bot (legacy compatibility)"""
    @bot.command(name="image", aliases=["fractal", "aion"])
    async def image_command(ctx, mode: str = "ouroboros"):
        """Generate AION fractal visualizations"""
        # Normalize mode
        mode = mode.lower()
        mode_aliases = {
            "aion": "ouroboros",
            "mandala": "ouroboros",
            "fractal": "mandelbrot",
        }
        mode = mode_aliases.get(mode, mode)

        if mode == "cycle":
            await ctx.send(
                "🔄 **Fractal Auto-Cycling Feature**\n"
                "⚠️ Not yet implemented - coming soon!"
            )
            return

        if mode not in ["ouroboros", "mandelbrot"]:
            await ctx.send(f"❌ Unknown mode: `{mode}`")
            return

        await ctx.send(f"🎨 **Generating AION {mode.upper()} fractal...**")

        try:
            ucf_state = load_ucf_state()
            result = await generate_and_post_to_discord(ucf_state, ctx.channel, mode)

            if result:
                await ctx.send(f"✅ **{mode.upper()} fractal complete!**")
            else:
                await ctx.send("❌ **Generation failed**")

        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")
