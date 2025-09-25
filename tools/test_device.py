from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixstu.tools.device import pick_device


if __name__ == "__main__":
    print("[PixStu] Device =", pick_device())
