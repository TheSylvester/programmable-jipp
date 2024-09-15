import pytest
import logging
import os
from jipp.utils.logging_utils import (
    setup_logger,
    get_logger,
    GlobalLoggerSettings,
    LoggerProxy,
)
from logging.handlers import RotatingFileHandler
import gzip
from jipp.utils.logging_utils import (
    compress_log,
    CompressedRotatingFileHandler,
)
import asyncio
import jipp.utils
import io
import sys


@pytest.fixture(autouse=True)
def reset_singletons():
    GlobalLoggerSettings._instance = None
    LoggerProxy._instance = None


@pytest.fixture
def temp_log_dir(tmp_path):
    """
    Creates a temporary directory for log files.

    :param tmp_path: Built-in pytest fixture providing a temporary path.
    :return: Path to the temporary log directory.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def test_setup_logger_basic(temp_log_dir):
    logger = setup_logger(name="test_logger", log_dir=str(temp_log_dir))
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG  # This should be DEBUG, not INFO
    assert len(logger.handlers) == 2  # Console and file handler


def test_setup_logger_console_level(temp_log_dir):
    logger = setup_logger(
        name="test_logger", console_level=logging.WARNING, log_dir=str(temp_log_dir)
    )
    console_handler = next(
        h for h in logger.handlers if isinstance(h, logging.StreamHandler)
    )
    assert console_handler.level == logging.WARNING


def test_setup_logger_file_handler(temp_log_dir):
    logger = setup_logger(name="test_logger", log_dir=str(temp_log_dir))
    file_handler = next(
        h for h in logger.handlers if isinstance(h, CompressedRotatingFileHandler)
    )
    assert isinstance(file_handler, CompressedRotatingFileHandler)
    expected_path = os.path.join(str(temp_log_dir), "test_logger.log")
    assert file_handler.baseFilename == expected_path


def test_setup_logger_max_bytes_and_backup_count(temp_log_dir):
    max_bytes = 1000
    backup_count = 3
    logger = setup_logger(
        name="test_logger",
        log_dir=str(temp_log_dir),
        max_bytes=max_bytes,
        backup_count=backup_count,
    )
    file_handler = next(
        h for h in logger.handlers if isinstance(h, CompressedRotatingFileHandler)
    )
    assert file_handler.maxBytes == max_bytes
    assert file_handler.backupCount == backup_count


def test_setup_logger_no_file_logging(temp_log_dir):
    logger = setup_logger(
        name="test_logger", log_dir=str(temp_log_dir), log_to_file=False
    )
    assert len(logger.handlers) == 1  # Only console handler
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_setup_logger_propagate(temp_log_dir):
    logger = setup_logger(name="test_logger", log_dir=str(temp_log_dir), propagate=True)
    assert logger.propagate == True


def test_get_logger():
    logger = get_logger("test_get_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_get_logger"


def test_compress_log(temp_log_dir):
    source_file = temp_log_dir / "test.log"
    dest_file = temp_log_dir / "test.log.gz"

    with open(source_file, "w") as f:
        f.write("Test log content")

    compress_log(str(source_file), str(dest_file))

    assert not os.path.exists(source_file)
    assert os.path.exists(dest_file)

    with gzip.open(dest_file, "rt") as f:
        content = f.read()
        assert content == "Test log content"


@pytest.mark.asyncio
async def test_logger_async_compatibility(temp_log_dir):
    logger = setup_logger(name="async_test_logger", log_dir=str(temp_log_dir))

    async def async_log_test():
        logger.info("Async logging test")

    await async_log_test()
    await asyncio.sleep(0.1)  # Add a small delay

    log_file = temp_log_dir / "async_test_logger.log"
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        content = f.read()
        assert "Async logging test" in content


def test_setup_logger_duplicate_calls(temp_log_dir):
    logger1 = setup_logger(name="duplicate_test", log_dir=str(temp_log_dir))
    logger2 = setup_logger(name="duplicate_test", log_dir=str(temp_log_dir))

    assert logger1 is logger2
    assert len(logger1.handlers) == 2  # Ensure handlers are not duplicated


def test_compressed_rotating_file_handler(temp_log_dir):
    log_file = temp_log_dir / "rotate_test.log"
    handler = CompressedRotatingFileHandler(str(log_file), maxBytes=50, backupCount=3)

    logger = logging.getLogger("rotate_test")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    for i in range(10):
        logger.debug("A" * 10)

    # Force a rollover
    handler.doRollover()

    assert os.path.exists(log_file)
    assert os.path.exists(str(log_file) + ".1.gz")  # Note the change here
    assert not os.path.exists(str(log_file) + ".4.gz")  # Should only keep 3 backups


def test_logger_proxy(monkeypatch, temp_log_dir):
    log = LoggerProxy()
    monkeypatch.setattr("jipp.utils.logging_utils.log", log)
    monkeypatch.setattr(
        "jipp.utils.logging_utils.setup_logger",
        lambda name, **kwargs: logging.getLogger(name),
    )

    # Force the module name to be "test_logging_utils"
    monkeypatch.setattr(
        LoggerProxy, "_get_module_name", lambda self: "test_logging_utils"
    )

    log.debug("Test message")

    assert "test_logging_utils" in log._loggers
    assert isinstance(log._loggers["test_logging_utils"], logging.Logger)


def test_logger_proxy_dynamic_console_level(monkeypatch):
    log = LoggerProxy()
    monkeypatch.setattr("jipp.utils.logging_utils.log", log)

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    log.debug("This should not appear")
    assert "This should not appear" not in captured_output.getvalue()

    log.console_level = "DEBUG"
    log.debug("This should appear")
    assert "This should appear" in captured_output.getvalue()

    log.console_level = "INFO"
    log.debug("This should not appear again")
    log.info("But this should appear")

    output = captured_output.getvalue()
    assert "This should not appear again" not in output
    assert "But this should appear" in output

    # Reset stdout
    sys.stdout = sys.__stdout__


def test_logger_proxy_per_module_console_level(monkeypatch):
    log = LoggerProxy()
    monkeypatch.setattr("jipp.utils.logging_utils.log", log)

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Test default behavior
    log.debug("This should not appear")
    assert "This should not appear" not in captured_output.getvalue()

    # Set console level for current module
    log.console_level = "DEBUG"
    log.debug("This should appear")
    assert "This should appear" in captured_output.getvalue()

    # Set console level for a different module
    log.set_console_level("ERROR", module="other_module")

    # This should still appear because we're in the current module
    log.debug("This should still appear")
    assert "This should still appear" in captured_output.getvalue()

    # Reset stdout
    sys.stdout = sys.__stdout__


def test_logger_proxy_global_console_level(monkeypatch):
    log = LoggerProxy()
    monkeypatch.setattr("jipp.utils.logging_utils.log", log)

    # Capture stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Set global console level
    log.set_global_console_level("DEBUG")

    # Force creation of a logger
    log.debug("This should appear")

    # Ensure the log message is flushed
    sys.stdout.flush()

    output = captured_output.getvalue()
    assert "This should appear" in output

    # Reset stdout
    sys.stdout = sys.__stdout__


def test_logger_proxy_console_level_property(monkeypatch):
    log = LoggerProxy()
    monkeypatch.setattr("jipp.utils.logging_utils.log", log)

    # Test default level
    assert log.console_level == "INFO"

    # Set and get level for current module
    log.console_level = "DEBUG"
    assert log.console_level == "DEBUG"

    # Set and get global level
    log.set_global_console_level("WARNING")
    assert log.console_level == "DEBUG"  # Should still be DEBUG for current module

    # Test level for a different module
    log.set_console_level("ERROR", module="other_module")
    assert log.console_level == "DEBUG"  # Should still be DEBUG for current module

    # Check console level for 'other_module'
    assert log.get_console_level(module_name="other_module") == "ERROR"

    # Check console level for a new module with no specific level set
    assert log.get_console_level(module_name="new_module") == "WARNING"  # Global level


def test_logger_proxy_global_console_level_behavior(monkeypatch):
    log = LoggerProxy()
    monkeypatch.setattr("jipp.utils.logging_utils.log", log)

    # Set global level
    log.set_global_console_level("ERROR")

    # Check level in a new module
    assert log.get_console_level(module_name="new_module") == "ERROR"

    # Set a module-specific level
    log.console_level = "DEBUG"

    # Check that the new module still uses the global level
    assert log.get_console_level(module_name="new_module") == "ERROR"

    # Check that the current module uses its specific level
    assert log.console_level == "DEBUG"
