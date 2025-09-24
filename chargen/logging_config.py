import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from .model_setup import LOGS

_DEFAULT_LEVEL = os.getenv("CHARGEN_LOG_LEVEL", "INFO").upper()
LOG_FILE = os.path.join(LOGS, "chargen.log")


def configure_logging(level: Optional[str] = None) -> str:
    desired_level = getattr(logging, (level or _DEFAULT_LEVEL), logging.INFO)
    root = logging.getLogger()
    if getattr(configure_logging, "_configured", False):
        root.setLevel(desired_level)
        return LOG_FILE

    os.makedirs(LOGS, exist_ok=True)

    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(desired_level)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    configure_logging._configured = True  # type: ignore[attr-defined]
    root.debug("Logging configured. Log file: %s", LOG_FILE)
    return LOG_FILE


def get_log_file() -> str:
    return LOG_FILE
