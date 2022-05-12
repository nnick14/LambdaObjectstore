import logging
from typing import Optional


def initialize_logger(add_handler: bool = False) -> logging.Logger:
    if add_handler:
        handler = set_file_handler("load_times.log")
    else:
        handler = None
    logger = get_logger("infinicache_logger", handler)
    return logger


def set_file_handler(filename: str) -> logging.FileHandler:
    handler = logging.FileHandler(filename)
    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    return handler


def get_logger(module_name: str, handler: Optional[logging.FileHandler] = None) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    if handler:
        logger.addHandler(handler)
    return logger
