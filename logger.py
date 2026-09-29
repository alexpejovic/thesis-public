import logging
import sys
from logging import Logger, NullHandler

import jax.numpy as jnp
from git import Repo
from jax import Array


def get_logger(filename: str, name: str) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(
        logging.DEBUG
    )  # Set to lowest level to allow all handlers to process

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Console only shows INFO and above
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)  # File logs everything
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_nil_logger() -> Logger:
    logger = logging.getLogger(__name__)
    logger.addHandler(NullHandler())
    return logger


def get_std_logger() -> Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def dict_to_str(d: dict[str, Array]) -> str:
    array_format = lambda a: float(a) if jnp.isscalar(a) else a
    return ", ".join(f"{k}={array_format(v)}" for k, v in d.items())


def log_values(logger: Logger, level: int, **values):
    """This is for using with jax.debug.callback()"""
    logger.log(int(level), dict_to_str(values))


def log_commit(logger: Logger, base_path: str):
    commit = Repo(base_path).commit()
    commit_hash = commit.hexsha
    log_str = f"Commit Hash: {commit_hash}"
    logger.info(log_str)
