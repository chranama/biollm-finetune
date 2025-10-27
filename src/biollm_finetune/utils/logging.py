# src/bioasq_llm/utils/logging.py
from __future__ import annotations
import logging
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

_console: Optional[Console] = None

def console() -> Console:
    global _console
    if _console is None:
        _console = Console()
    return _console

def get_logger(name: str = "bioasq_llm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = RichHandler(console=console(), show_time=False, markup=True, rich_tracebacks=True)
    fmt = logging.Formatter("%(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger