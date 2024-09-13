import pytest
import logging
import os
import gzip
from utils.logging_utils import (
    setup_logger,
    compress_log,
    CompressedRotatingFileHandler,
)
import asyncio


@pytest.fixture
def temp_log_dir(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
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
    assert file_handler.baseFilename == str(temp_log_dir / "test_logger.log")


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
