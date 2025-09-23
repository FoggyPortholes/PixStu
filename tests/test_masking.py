from PIL import Image

from app.masking import estimate_bg_color, make_alpha


def test_estimate_bg_color_simple():
    img = Image.new("RGB", (10, 10), (10, 20, 30))
    assert estimate_bg_color(img) == (10, 20, 30)


def test_make_alpha_tolerance():
    img = Image.new("RGB", (4, 4), (0, 255, 0))
    img.putpixel((1, 1), (0, 250, 0))  # close to bg
    out = make_alpha(img, (0, 255, 0), tol=10)
    assert out.getpixel((1, 1))[3] == 0  # transparent pixel
