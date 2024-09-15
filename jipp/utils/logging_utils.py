import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
import sys
import gzip
import shutil
from typing import Union, Optional
import inspect

# Global configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "simple": {"format": "%(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console"]},
}

try:
    logging.config.dictConfig(LOGGING_CONFIG)
except Exception as e:
    print(f"Failed to configure global logging: {str(e)}", file=sys.stderr)


def compress_log(source, dest):
    with open(source, "rb") as f_in:
        with gzip.open(f"{dest}", "wb") as f_out:  # Remove the .gz extension here
            shutil.copyfileobj(f_in, f_out)
    os.remove(source)


class CompressedRotatingFileHandler(RotatingFileHandler):
    def rotation_filename(self, default_name):
        return default_name + ".gz"

    def rotate(self, source, dest):
        with open(source, "rb") as f_in:
            with gzip.open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(source)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d" % (self.baseFilename, i + 1))
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self.rotation_filename(self.baseFilename + ".1")
            if os.path.exists(dfn):
                os.remove(dfn)
            self.rotate(self.baseFilename, dfn)
        if not self.delay:
            self.stream = self._open()


def setup_logger(
    name: Optional[str] = None,
    console_level: Union[str, int] = None,  # Changed from "INFO" to None
    log_dir: str = "logs",
    max_bytes: int = 1_000_000,  # 1 MB
    backup_count: int = 5,
    log_to_file: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """
    Sets up and returns a logger with console logging at specified level and optional file logging.

    :param name: The name of the logger, defaults to the module name.
    :param console_level: The logging level for console output as string or int (e.g., "DEBUG", "INFO", logging.WARNING).
    :param log_dir: Directory to store log files.
    :param max_bytes: The maximum size (in bytes) for the log file before rotating.
    :param backup_count: The number of backup log files to keep after rotating.
    :param log_to_file: Whether to log to a file in addition to console.
    :param propagate: Whether the logger should propagate messages to its parent.
    :return: Configured logger instance.
    """
    name = name or __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture all logs
    logger.propagate = propagate

    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Use global console level if not specified
    if console_level is None:
        console_level = GlobalLoggerSettings().global_console_level

    # Convert string level to int if necessary
    if isinstance(console_level, str):
        console_level = getattr(logging, console_level.upper())

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name.replace('.', '_')}.log")
            file_handler = CompressedRotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setLevel(logging.DEBUG)  # Capture all log levels in file
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except (IOError, OSError) as e:
            logger.error(f"Failed to set up file logging: {str(e)}")
            logger.warning("Continuing with console logging only.")
        except Exception as e:
            logger.error(f"Unexpected error setting up file logging: {str(e)}")
            logger.warning("Continuing with console logging only.")

    return logger


# Convenience function to get a logger
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the given name. If the logger doesn't exist, it will be created with default settings.

    :param name: The name of the logger.
    :return: Logger instance.
    """
    return logging.getLogger(name or __name__)


# Global configuration
try:
    logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG to allow all log levels
except Exception as e:
    print(f"Failed to configure global logging: {str(e)}", file=sys.stderr)


class GlobalLoggerSettings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._global_console_level = (
                logging.INFO
            )  # Ensure this is INFO by default
        return cls._instance

    @property
    def global_console_level(self):
        return self._global_console_level

    @global_console_level.setter
    def global_console_level(self, level: Union[str, int]):
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self._global_console_level = level


class LoggerProxy:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loggers = {}
            cls._instance._console_levels = {}
            cls._instance._global_settings = GlobalLoggerSettings()
        return cls._instance

    def __getattr__(self, name):
        module_name = self._get_module_name()
        return self._get_logger_attr(module_name, name)

    def _get_module_name(self):
        caller_frame = inspect.currentframe().f_back
        return caller_frame.f_globals.get("__name__")

    def _get_logger_attr(self, module_name, attr_name):
        if module_name not in self._loggers:
            console_level = self._console_levels.get(
                module_name, self._global_settings.global_console_level
            )
            self._loggers[module_name] = setup_logger(
                name=module_name, console_level=console_level
            )
        return getattr(self._loggers[module_name], attr_name)

    def get_console_level(self, module_name=None):
        if module_name is None:
            module_name = self._get_module_name()
        level = self._console_levels.get(
            module_name, self._global_settings.global_console_level
        )
        return logging.getLevelName(level)

    @property
    def console_level(self):
        module_name = self._get_module_name()
        return self.get_console_level(module_name)

    @console_level.setter
    def console_level(self, level):
        module_name = self._get_module_name()
        self.set_console_level(level, module=module_name)

    def set_console_level(self, level, module=None):
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        if module is None:
            module = self._get_module_name()
        self._console_levels[module] = level
        if module in self._loggers:
            self._update_logger_level(module, level)

    def set_global_console_level(self, level):
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self._global_settings.global_console_level = level
        self._update_all_logger_levels()

    def _update_logger_level(self, module_name: str, level: int):
        logger = self._loggers[module_name]
        logger.setLevel(level)
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(level)

    def _update_all_logger_levels(self):
        for module_name, logger in self._loggers.items():
            level = self._console_levels.get(
                module_name, self._global_settings.global_console_level
            )
            self._update_logger_level(module_name, level)


log = LoggerProxy()

# Note: Default logging levels
# - Console: INFO (can be changed using log.console_level setter)
# - File: DEBUG (captures all log levels in file, if file logging is enabled)

# TODO: Convert print() statements to appropriate log calls
# - Use log.info() for general information (equivalent to most print() statements)
# - Use log.debug() for detailed debug information
# - Use log.warning() for warning messages
# - Use log.error() for error messages
# - Use log.critical() for critical errors that may cause the application to fail

# Example usage:
# log.info("Application started")
# log.debug("Detailed debug information")
# log.warning("Warning: resource running low")
# log.error("An error occurred: {}".format(error_message))
# log.critical("Critical error: application shutting down")
