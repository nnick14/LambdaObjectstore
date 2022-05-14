import logging
import sys
from typing import Optional

DATALOG_LOAD_TRAINING = 10
DATALOG_LOAD_VALIDATION = 11
DATALOG_LOAD_ALL = 21
DATALOG_TRAINING = 0
DATALOG_VALIDATION = 1
DATALOG_BATCH = 2

def initialize_logger(add_handler: bool = False) -> logging.Logger:
    if add_handler:
        handler = set_std_handler()
    else:
        handler = None
    
    logger = get_logger("default", handler)
    return logger


def set_file_handler(filename: str) -> logging.FileHandler:
    handler = logging.FileHandler(filename)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    return handler

def set_std_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    return handler

def get_logger(module_name: str, handler: Optional[logging.FileHandler] = None) -> logging.Logger:
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    if handler is not None:
        logger.addHandler(handler)
    return logger
