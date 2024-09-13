import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(
    name=None,
    console_level=logging.INFO,
    log_dir="logs",
    max_bytes=1_000_000,  # 1 MB
    backup_count=5,
):
    """
    Sets up and returns a logger with console logging at specified level and file logging for all levels.

    :param name: The name of the logger, defaults to the module name.
    :param console_level: The logging level for console output (e.g., DEBUG, INFO, WARNING).
    :param log_dir: Directory to store log files.
    :param max_bytes: The maximum size (in bytes) for the log file before rotating.
    :param backup_count: The number of backup log files to keep after rotating.
    :return: Configured logger instance.
    """
    name = name or __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs

    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name.replace('.', '_')}.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)  # Capture all log levels in file
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
