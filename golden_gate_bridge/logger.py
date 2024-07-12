"""Code from `/openai_whisper/finetuning/train/logs.py`."""

import logging


def get_logger(
    name: str | None = None, level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_logging(*, log_level: int) -> None:
    import datasets
    import transformers

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
