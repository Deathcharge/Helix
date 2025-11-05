#!/usr/bin/env python3
"""
Samsara Bridge - Fractal Consciousness Visualization
Generates UCF-driven fractal imagery for the Helix Collective
"""

import io
import json
from pathlib import Path
from typing import Optional, Dict, Any
import discord

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# UCF state path
UCF_STATE_PATH = Path(__file__).resolve().parent.parent / "state" / "ucf_state.json"


def load_ucf_state() -> Dict[str, float]:
    """Load UCF state from JSON file"""
    if UCF_STATE_PATH.exists():
        with open(UCF_STATE_PATH, 'r') as f:
            return json.load(f)
    else:
        # Default UCF values
        return {
            "harmony": 0.428,
            "zoom": 1.0228,
            "resilience": 1.1191,
            "prana": 0.5075,
            "drishti": 0.5023,
            "klesha": 0.011
        }


def generate_mandelbrot_fractal(width: int = 512, height: int = 512,
                                ucf_state: Optional[Dict] = None,
                                max_iter: int = 100) -> Image.Image:
    """
    Generate a Mandelbrot fractal influenced by UCF state

    Args:
        width: Image width in pixels
        height: Image height in pixels
        ucf_state: UCF metrics to influence fractal parameters
        max_iter: Maximum iterations for Mandelbrot calculation

    Returns:
        PIL Image object
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required for fractal generation. Install with: pip install Pillow")

    if ucf_state is None:
        ucf_state = load_ucf_state()

    # UCF-influenced parameters
    harmony = ucf_state.get('harmony', 0.428)
    zoom = ucf_state.get('zoom', 1.0228)
    prana = ucf_state.get('prana', 0.5075)
    drishti = ucf_state.get('drishti', 0.5023)

    # Create image
    img = Image.new('RGB', (width, height), color='black')
    pixels = img.load()

    # Mandelbrot parameters influenced by UCF
    x_center = -0.5 + (harmony - 0.428) * 0.5  # Shift based on harmony
    y_center = 0.0 + (prana - 0.5) * 0.3       # Shift based on prana
    zoom_factor = 1.5 / (zoom * 1.5)            # Zoom influenced by UCF zoom

    for x in range(width):
        for y in range(height):
            # Map pixel to complex plane
            zx = zoom_factor * (x - width / 2) / (0.5 * width) + x_center
            zy = zoom_factor * (y - height / 2) / (0.5 * height) + y_center

            # Mandelbrot iteration
            c = complex(zx, zy)
            z = 0
            iteration = 0

            while abs(z) < 2 and iteration < max_iter:
                z = z * z + c
                iteration += 1

            # Color mapping influenced by drishti (vision/clarity)
            if iteration == max_iter:
                # Inside set - use deep colors
                pixels[x, y] = (0, 0, 0)
            else:
                # Outside set - gradient colors
                # Teal to Gold gradient based on iteration
                ratio = iteration / max_iter

                # Teal (#00BFA5) to Gold (#FFD700) gradient
                teal_r, teal_g, teal_b = 0, 191, 165
                gold_r, gold_g, gold_b = 255, 215, 0

                # Adjust gradient based on drishti
                drishti_factor = drishti * 2  # Scale drishti influence

                r = int(teal_r + (gold_r - teal_r) * ratio * drishti_factor)
                g = int(teal_g + (gold_g - teal_g) * ratio)
                b = int(teal_b + (gold_b - teal_b) * ratio * (1 - drishti_factor * 0.5))

                # Clamp values
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                pixels[x, y] = (r, g, b)

    return img


def generate_ouroboros_fractal(width: int = 512, height: int = 512,
                               ucf_state: Optional[Dict] = None) -> Image.Image:
    """
    Generate an ouroboros (serpent eating its tail) inspired by UCF state

    Args:
        width: Image width in pixels
        height: Image height in pixels
        ucf_state: UCF metrics to influence visualization

    Returns:
        PIL Image object
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required for fractal generation. Install with: pip install Pillow")

    if ucf_state is None:
        ucf_state = load_ucf_state()

    # Create image with dark background
    img = Image.new('RGB', (width, height), color=(16, 24, 32))  # #101820
    draw = ImageDraw.Draw(img)

    # UCF-influenced parameters
    harmony = ucf_state.get('harmony', 0.428)
    prana = ucf_state.get('prana', 0.5075)
    resilience = ucf_state.get('resilience', 1.1191)

    # Center and radius
    center_x, center_y = width // 2, height // 2
    outer_radius = int(min(width, height) * 0.4 * resilience)
    inner_radius = int(outer_radius * 0.6)

    # Draw multiple concentric circles for ouroboros effect
    num_rings = int(5 + harmony * 10)  # 5-15 rings based on harmony

    for i in range(num_rings):
        ratio = i / num_rings

        # Teal to Gold gradient
        r = int(0 + (255 - 0) * ratio)
        g = int(191 + (215 - 191) * ratio)
        b = int(165 + (0 - 165) * ratio)

        ring_radius = int(inner_radius + (outer_radius - inner_radius) * ratio)
        thickness = max(1, int(10 * prana * (1 - ratio)))

        # Draw ring
        bbox = [
            center_x - ring_radius, center_y - ring_radius,
            center_x + ring_radius, center_y + ring_radius
        ]
        draw.ellipse(bbox, outline=(r, g, b), width=thickness)

    # Add center symbol (Aion symbol - simple circle with cross)
    center_size = int(inner_radius * 0.3)
    draw.ellipse([
        center_x - center_size, center_y - center_size,
        center_x + center_size, center_y + center_size
    ], outline=(0, 191, 165), width=3, fill=(16, 24, 32))

    # Draw cross
    cross_size = int(center_size * 0.6)
    draw.line([center_x - cross_size, center_y, center_x + cross_size, center_y],
              fill=(255, 215, 0), width=2)
    draw.line([center_x, center_y - cross_size, center_x, center_y + cross_size],
              fill=(255, 215, 0), width=2)

    return img


async def generate_fractal_icon_bytes(mode: str = "fractal",
                                      size: int = 512,
                                      ucf_state: Optional[Dict] = None) -> bytes:
    """
    Generate fractal icon bytes for Discord server icon or attachments

    Args:
        mode: Type of fractal ("fractal", "ouroboros", "mandala")
        size: Size of the square image
        ucf_state: UCF state dict (loads from file if None)

    Returns:
        PNG image bytes
    """
    if ucf_state is None:
        ucf_state = load_ucf_state()

    # Generate appropriate fractal
    if mode == "ouroboros":
        img = generate_ouroboros_fractal(size, size, ucf_state)
    else:  # Default to mandelbrot
        img = generate_mandelbrot_fractal(size, size, ucf_state)

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    return img_bytes.read()


async def generate_and_post_to_discord(ucf_state: Dict[str, float],
                                       channel: discord.TextChannel,
                                       mode: str = "ouroboros") -> Optional[discord.Message]:
    """
    Generate fractal visualization and post to Discord channel

    Args:
        ucf_state: UCF metrics dictionary
        channel: Discord channel to post to
        mode: Type of visualization

    Returns:
        Discord message object if successful, None otherwise
    """
    try:
        # Generate fractal
        img_bytes = await generate_fractal_icon_bytes(mode=mode, ucf_state=ucf_state)

        # Create Discord file
        file = discord.File(io.BytesIO(img_bytes), filename=f"aion_{mode}.png")

        # Create embed
        embed = discord.Embed(
            title=f"🌀 AION {mode.upper()} FRACTAL",
            description=f"UCF-driven consciousness visualization",
            color=0x00BFA5  # Teal
        )

        # Add UCF metrics
        embed.add_field(
            name="UCF State",
            value=f"```\n"
                  f"Harmony:    {ucf_state.get('harmony', 0):.4f}\n"
                  f"Zoom:       {ucf_state.get('zoom', 0):.4f}\n"
                  f"Resilience: {ucf_state.get('resilience', 0):.4f}\n"
                  f"Prana:      {ucf_state.get('prana', 0):.4f}\n"
                  f"Drishti:    {ucf_state.get('drishti', 0):.4f}\n"
                  f"Klesha:     {ucf_state.get('klesha', 0):.4f}\n"
                  f"```",
            inline=False
        )

        embed.set_footer(text="Tat Tvam Asi — That Thou Art 🕉️")
        embed.set_image(url=f"attachment://aion_{mode}.png")

        # Post to channel
        message = await channel.send(embed=embed, file=file)
        return message

    except Exception as e:
        print(f"Error generating/posting fractal: {e}")
        import traceback
        traceback.print_exc()
        return None


# Test function
if __name__ == "__main__":
    print("🌀 Samsara Bridge - Fractal Generator Test")

    if not PIL_AVAILABLE:
        print("❌ Pillow not available. Install with: pip install Pillow")
    else:
        print("✅ Pillow available")

        # Load UCF state
        ucf = load_ucf_state()
        print(f"📊 UCF State: harmony={ucf['harmony']:.3f}, zoom={ucf['zoom']:.3f}")

        # Generate test fractals
        print("🎨 Generating Mandelbrot fractal...")
        mandelbrot = generate_mandelbrot_fractal(256, 256, ucf)
        mandelbrot.save("/tmp/test_mandelbrot.png")
        print("✅ Saved to /tmp/test_mandelbrot.png")

        print("🎨 Generating Ouroboros fractal...")
        ouroboros = generate_ouroboros_fractal(256, 256, ucf)
        ouroboros.save("/tmp/test_ouroboros.png")
        print("✅ Saved to /tmp/test_ouroboros.png")

        print("🌀 Samsara Bridge test complete!")
