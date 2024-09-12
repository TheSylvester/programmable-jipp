import logging
from logging.handlers import RotatingFileHandler


def setup_logger(
    name=__name__, level=logging.INFO, log_file=None, max_bytes=2000, backup_count=5
):
    """
    Sets up and returns a logger for the given module or class with optional file logging and log rotation.

    The log file will be rotated when it exceeds the max_bytes size, and up to `backup_count` backup log files will be kept.

    :param name: The name of the logger, typically the module or class name.
    :param level: The logging level (e.g., DEBUG, INFO, WARNING).
    :param log_file: Optional file path for saving logs. If None, logs will only go to the console.
    :param max_bytes: The maximum size (in bytes) for the log file before rotating.
    :param backup_count: The number of backup log files to keep after rotating.
    :return: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional rotating file handler
        if log_file:
            # Create a rotating file handler
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
