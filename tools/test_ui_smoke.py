#!/usr/bin/env python3
"""
Smoke test for PixStu UI — ensures Studio builds without crashing.
"""
from chargen import studio


def test_ui():
    app = studio.studio()
    assert app is not None
    print("[SMOKE] Studio UI built successfully")


if __name__ == "__main__":
    test_ui()
