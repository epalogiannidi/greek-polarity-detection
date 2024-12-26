"""The structure of this project consists of:

- -a **brain.py** that is the core of the project
    - -Contains the main functionality
    - -imports and employs supportive modules or packages
- -a **utils** package that contains supportive functionality
- -an **about.py** that contains the project details

Project supports logging and a global logger is initialized in the main package.

Example of using logger
-----------------------
Import:
    .. highlight:: python
    .. code-block:: python

        from src import logger
Usage:
    .. highlight:: python
    .. code-block:: python

        logger.info("info message")
        logger.debug("debug message")
        logger.error("error message")

For all the logging messages look at:
    https://docs.python.org/3.8/howto/logging.html
"""

from greek_polarity_detection.about import __title__  # noqa F401
from greek_polarity_detection.log_setup import Logger  # noqa F401
from greek_polarity_detection.utils import (
    Dataset,
    Lexicon,  # noqa F401
    LexiconTrainer,
    SetUpConfig,
)

logger = Logger().logger
