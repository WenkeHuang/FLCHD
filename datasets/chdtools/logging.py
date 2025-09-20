import logging
from rich.logging import RichHandler


def get_logger(name: str, level=logging.INFO, rich_handler: bool = True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if rich_handler:
        handler = RichHandler(rich_tracebacks=True, enable_link_path=False)
    else:
        handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s]%(message)s",
            datefmt="%y/%m/%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
