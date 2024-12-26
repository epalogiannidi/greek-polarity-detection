""" Contains the project details,
    e.g., *title*, *version*, *summary* etc.
"""

__MAJOR__ = 0
__MINOR__ = 0
__PATCH__ = 1

__title__ = "greek_polarity_detection"
__version__ = ".".join([str(__MAJOR__), str(__MINOR__), str(__PATCH__)])
__summary__ = "Greek polarity detection methods using classic affective lexicon based methods and LLMs."
__author__ = "Elisavet Palogiannidi"
__copyright__ = f"Copyright (C) 2024 {__author__}"
__email__ = "epalogiannidi@gmail.com"


if __name__ == "__main__":
    print(__version__)
