from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixstu.app.studio import studio


def test_ui():
    demo = studio()
    assert demo is not None
    if hasattr(demo, "close"):
        demo.close()
    print("[SMOKE] UI OK")


if __name__ == "__main__":
    test_ui()
