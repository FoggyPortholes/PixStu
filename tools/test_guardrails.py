import pytest
from pixstu.tools.guardrails import check_prompt, check_blank_background
from PIL import Image


def test_prompt_forbidden():
    try:
        check_prompt("pixel art comic strip")
    except ValueError:
        assert True


def test_blank_background():
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    check_blank_background(img)
