import os
import logging.config
from logging import getLogger

import yaml

from span_api_splitter import literals


def load_config(path):
    """
    Loads the config from yaml file.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

log_file = load_config('configs/logging.yaml')
logging.config.dictConfig(log_file)
logger = getLogger(literals.LOGGER_NAME)


config = load_config('configs/span_api_splitter_dev.yaml')