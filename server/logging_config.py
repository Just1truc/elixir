"""Central logging configuration for Zappy.

Usage:
    from server.logging_config import configure_logging, get_logger
    configure_logging()  # once at app start
    logger = get_logger(__name__)
    logger.info("message")
"""

from __future__ import annotations

import logging
import logging.config
import os
from typing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging for the application.
    Level can be overridden via argument or LOG_LEVEL env var.
    Safe to call multiple times (idempotent).
    """
    log_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
