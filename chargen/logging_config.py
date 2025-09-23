import logging
import os
from typing import Optional

_DEFAULT_LEVEL = os.getenv("CHARGEN_LOG_LEVEL", "INFO").upper()


def configure_logging(level: Optional[str] = None) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=getattr(logging, (level or _DEFAULT_LEVEL), logging.INFO),
        format="[%(levelname)s] %(name)s: %(message)s",
    )
