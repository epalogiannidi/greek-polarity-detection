## MAKEFILE

.PHONY: help
help:
	@echo "Available targets:"
	@echo "	environment		: create a virtual environment and install dependencies"
	@echo " change-origin	: change the origin with the origin of the new repo and clears template commit history"
	@echo "	black			: format code using black, the Python code formatter"
	@echo "	black-check		: check if source code complies with black"
	@echo "	lint			: check source code with flake8"
	@echo "	mypy			: check static typing with mypy"
	@echo "	check-codestyle	: perform a complete codestyle checking"
	@echo "	test			: run unit tests"
	@echo "	docs			: generate documentation"
	@echo " build-source    : build the source distribution (platform agnostic)"
	@echo " build-binary    : build the binary distribution (platform specific and faster)"
	@echo " build           : build source and binary distribution"
	@echo " clean           : clean various folders and files"
.PHONY: black
black:
	black recipe_package_template/

.PHONY: black-check
black-check:
	black --check recipe_package_template/

.PHONY: lint
lint:
	flake8 --max-line-length 120 --ignore E203,E402,W503 recipe_package_template/

.PHONY: mypy
mypy:
	mypy --config-file configs/mypi.ini recipe_package_template/
	rm -rf ./mypy_cache

.PHONY: check-codestyle
check-codestyle: black-check lint mypy

.PHONY: docs
docs:
	find docs/source/ -name "*.rst" | grep -v index.rst | xargs rm
	sphinx-apidoc -f -o docs/source/ recipe_package_template/
	cd docs && make clean && make html

.PHONY: test
test:
	pytest --cov-report term-missing --cov=recipe_package_template  --cov-config=.coveragerc tests/

	sleep 5
	rm -rf .pytest_cache/
	rm -f .coverage
	rm -f .coverage.*

.PHONY: environment
environment:
	pipenv --python 3.10
	pipenv install --dev

.PHONY: change-origin
change-origin:
	git remote set-url origin $(NEW_URL)
	# clear template history
	git checkout --orphan new_branch_name
	git add -A
	git commit -am "Initial commit"
	git branch -D main
	git branch -m main
	git push -f origin main

.PHONY: build-source
build-source:
	rm -rf build
	rm -rf recipe_package_template.egg-info/
	python setup.py bdist_wheel
	rm -rf build
	rm -rf recipe_package_template.egg-info/

.PHONY: build-binary
build-binary:
	rm -rf build
	rm -rf recipe_package_template.egg-info/
	echo "yes" | python setup.py bdist_nuitka > tmp.out
	rm tmp.out
	rm -rf build
	rm -rf recipe_package_template.egg-info/

.PHONY: build
build: build-source build-binary

PHONY: clean
clean:
	# clean up all the mess
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf dist
	rm -rf build
	rm -rf recipe_package_template.egg-info/