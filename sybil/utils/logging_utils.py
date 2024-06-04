import logging
import os


LOGGER_NAME = "sybil"
LOGLEVEL_KEY = "LOG_LEVEL"


def _get_formatter(loglevel="INFO"):
    warn_fmt = "[%(asctime)s] %(levelname)s - %(message)s"
    debug_fmt = "[%(asctime)s] [%(filename)s:%(lineno)d] %(levelname)s - %(message)s"
    fmt = debug_fmt if loglevel.upper() in {"DEBUG"} else warn_fmt
    return logging.Formatter(
        fmt=fmt,
        datefmt="%Y-%b-%d %H:%M:%S %Z",
    )


def remove_all_handlers(logger):
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


def configure_logger(loglevel=None, logger_name=LOGGER_NAME, logfile=None):
    """Do basic logger configuration and set our main logger"""

    # Set as environment variable so other processes can retrieve it
    if loglevel is None:
        loglevel = os.environ.get(LOGLEVEL_KEY, "WARNING")
    else:
        os.environ[LOGLEVEL_KEY] = loglevel

    logger = logging.getLogger(logger_name)
    logger.setLevel(loglevel)
    remove_all_handlers(logger)
    logger.propagate = False

    formatter = _get_formatter(loglevel)
    def _prep_handler(handler):
        for ex_handler in logger.handlers:
            if type(ex_handler) == type(handler):
                # Remove old handler, don't want to double-handle
                logger.removeHandler(ex_handler)
        handler.setLevel(loglevel)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    sh = logging.StreamHandler()
    _prep_handler(sh)

    if logfile is not None:
        fh = logging.FileHandler(logfile, mode="a")
        _prep_handler(fh)

    return logger


def get_logger(base_name=LOGGER_NAME):
    """
    Return a logger.
    Use a different logger in each subprocess, though they should all have the same log level.
    """
    pid = os.getpid()
    logger_name = f"{base_name}-process-{pid}"
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        configure_logger(logger_name=logger_name)
    return logger
