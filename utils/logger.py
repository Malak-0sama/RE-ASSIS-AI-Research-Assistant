"""
Structured logging setup built on Loguru.

Provides a single `get_logger` factory used across every module so log
format, rotation, and destinations stay consistent throughout the app.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _loguru_logger

from config.settings import get_settings

_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_cfg = settings.app.logging

    log_dir = Path(log_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _loguru_logger.remove()

    _loguru_logger.add(
        sys.stderr,
        level=log_cfg.level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    _loguru_logger.add(
        log_dir / "research_ai.log",
        level=log_cfg.level,
        rotation=log_cfg.rotation,
        retention=log_cfg.retention,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    _CONFIGURED = True


def get_logger(name: str):
    """Return a Loguru logger bound with a module name for context."""
    _configure_once()
    return _loguru_logger.bind(module=name)
