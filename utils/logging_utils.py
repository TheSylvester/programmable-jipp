import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
import sys
import gzip
import shutil


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

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
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


# Global configuration
try:
    logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG to allow all log levels
except Exception as e:
    print(f"Failed to configure global logging: {str(e)}", file=sys.stderr)
