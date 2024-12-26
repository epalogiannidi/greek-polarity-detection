## MAKEFILE

.PHONY: help
help:
	@echo "Available targets:"
	@echo "	environment		: create a virtual environment and install dependencies"
	@echo " change-origin	: change the origin with the origin of the new repo and clears template commit history"
	@echo "	black			: format code using black, the Python code formatter"
	@echo "	lint			: check source code with flake8"
	@echo "	mypy			: check static typing with mypy"
	@echo "	isort			: sorts the imports"
	@echo "	test			: run unit tests"
	@echo "	docs			: generate documentation"

.PHONY: black
black:
	black greek_polarity_detection/

.PHONY: lint
lint:
	flake8 --max-line-length 120 --ignore E203,E402,W503 greek_polarity_detection/

.PHONY: mypy
mypy:
	mypy --config-file configs/mypi.ini greek_polarity_detection/
	rm -rf ./mypy_cache

.PHONY: isort
isort:
	isort greek_polarity_detection/