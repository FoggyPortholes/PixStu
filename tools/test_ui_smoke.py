#!/usr/bin/env python3
from chargen import studio

def test_ui():
    app = studio.studio(); assert app is not None
    print("[SMOKE] Studio UI built successfully")

if __name__ == "__main__": test_ui()
