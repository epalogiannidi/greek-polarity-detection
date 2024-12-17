import configparser
import os
from setuptools import find_packages, setup
from typing import List, Dict

from recipe_package_template.about import __title__, __version__, __summary__, __author__

# Available options: https://pypi.org/search/

__NAME__: str = __title__
__LICENSE_DESCRIPTION__: str = "Proprietary"
__PYTHON_REQUIRES__: str = "==3.10.*"
__CLASSIFIERS__: List[str] = [
    "Development Status :: 1 - Planning"
    "Intended Audience :: Developers"
    "Intended Audience :: Science/Research"
    "License :: Other/Proprietary/License"
    "Operating System :: OS Independent"
    "Programming Language :: Python :: 3"
    "Programming Language :: Python :: 3.10"
    "Programming Language :: Python :: Implementation :: CPython"
    "Topic :: Scientific/Engineering :: Artificial Intelligence"
    "Typing :: Typed",
    "Natural Language :: English"
]

__INSTALL_REQUIRES__: List[str] = []
config = configparser.ConfigParser(strict=False)
config.read("Pipfile")
for dep in config["packages"]:
    version = config["packages"][dep][1:-1]
    if version[0] in {"=", ">", "<"}:
        __INSTALL_REQUIRES__.append(dep + version)
    else:
        __INSTALL_REQUIRES__.append(dep)

__ENTRY_POINTS__: Dict[str, List[str]] = {
    "console_scripts": ["recipe-package-template=recipe_package_template.__main__:main"]
}

__PACKAGE_EXCLUDES__: List[str] = ["*.tests", "*.tests.*", "tests.*", "tests"]

if __name__ == "__main__":
    setup(
        name=__NAME__,
        description=__summary__,
        long_description=open(os.path.join(os.path.dirname(__file__), "README.md")).read(),
        version=__version__,
        author=__author__,
        license=__LICENSE_DESCRIPTION__,
        packages=find_packages(exclude=__PACKAGE_EXCLUDES__),
        install_requires=__INSTALL_REQUIRES__,
        python_requires=__PYTHON_REQUIRES__,
        classifiers=__CLASSIFIERS__,
        # command_options={"nuitka": {"no-pyi-file": None}},
        entry_points=__ENTRY_POINTS__
    )
