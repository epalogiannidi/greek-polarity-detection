""" Contains the project details,
    e.g., *title*, *version*, *summary* etc.
"""

__MAJOR__ = 0
__MINOR__ = 0
__PATCH__ = 1

__title__ = "recipe_package_template"
__version__ = ".".join([str(__MAJOR__), str(__MINOR__), str(__PATCH__)])
__summary__ = (
    "The recipe repo to bootstrap new projects that do not require to be build."
)
__author__ = "Elisavet Palogiannidi"
__copyright__ = f"Copyright (C) 2023 {__author__}"
__email__ = "epalogiannidi@gmail.com"


if __name__ == "__main__":
    print(__version__)
