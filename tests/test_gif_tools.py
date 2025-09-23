import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.gif_tools import save_gif, save_sprite_sheet, nn_resize


def test_nn_resize_crisp():
    img = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
    up = nn_resize(img, (32, 32))
    assert up.size == (32, 32)


def test_save_gif(tmp_path, monkeypatch):
    monkeypatch.setenv("PCS_OUTPUTS_DIR", str(tmp_path))
    from importlib import reload
    import app.gif_tools as gif_tools
    reload(gif_tools)

    frames = [Image.new("RGBA", (16, 16), (i * 10 % 255, 0, 0, 255)) for i in range(4)]
    out = gif_tools.save_gif(frames, duration_ms=80)
    assert out.endswith(".gif")
    assert os.path.exists(out)


def test_save_sprite_sheet(tmp_path, monkeypatch):
    monkeypatch.setenv("PCS_OUTPUTS_DIR", str(tmp_path))
    from importlib import reload
    import app.gif_tools as gif_tools
    reload(gif_tools)

    frames = [Image.new("RGBA", (16, 16), (0, i * 10 % 255, 0, 255)) for i in range(6)]
    out = gif_tools.save_sprite_sheet(frames, columns=3, padding=1)
    assert out.endswith(".png")
    assert os.path.exists(out)
