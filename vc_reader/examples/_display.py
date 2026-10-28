"""Small Pillow contact sheet for notebook figures; no reader/training logic."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def panel(items, *, size=256, vmin=0., vmax=1., font_size=10):
    """Display 2D arrays with the same explicit intensity range and captions."""
    canvas = Image.new('RGB', (size * len(items), size + 48), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, (title, values) in enumerate(items):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError('choose a 2D slice explicitly')
        gray = (np.clip((values-vmin) / max(vmax-vmin, 1e-8), 0, 1)*255).astype(np.uint8)
        tile = Image.fromarray(gray).resize((size, size), Image.Resampling.NEAREST)
        canvas.paste(tile, (column*size, 48))
        draw.multiline_text((column*size+8, 8), title, fill='black', spacing=4,
                            font=ImageFont.load_default(size=font_size))
    return canvas
