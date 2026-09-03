import logging
from logging import getLogger

from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=False,
            markup=True,
        )
    ],
)

logger = getLogger("helixworld_runtime")
logger.setLevel(logging.DEBUG)
logger.propagate = True

# Expose common logging functions directly
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical
