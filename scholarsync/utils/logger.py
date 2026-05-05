"""
Structured logging configuration for ScholarSync.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        # On Windows, sys.stdout may use cp1252 which cannot encode Unicode
        # characters (emojis, box-drawing, arrows, etc.) used in log messages.
        # Wrap stdout in a UTF-8 stream with 'replace' error handling so
        # logging never crashes due to encoding issues.
        import io
        import os

        if os.name == "nt":
            stream = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        else:
            stream = sys.stdout

        handler = logging.StreamHandler(stream)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
