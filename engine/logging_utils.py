from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _resolve_level(raw: str | None) -> int:
    name = str(raw or os.environ.get("PHANTOM_LOG_LEVEL", "INFO")).upper().strip()
    return int(getattr(logging, name, logging.INFO))


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger = logging.getLogger("engine")
    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    if logger.handlers:
        _CONFIGURED = True
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
    _CONFIGURED = True
