from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path

from app.core.config import PROJECT_ROOT


LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "app.log"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("photoselect")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info("程序启动")
    logger.info("Python: %s", sys.version.replace("\n", " "))
    logger.info("System: %s %s", platform.system(), platform.release())
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("photoselect")
