# PACKAGE BUILD

## Upload build to PyPi
archive:
	@ python3 setup.py test_upload
	@ # @ python3 -m twine upload --repository testpypi dist/* --verbose

## Install the package locally
## ONLY AFTER RUNNING `make build`
local-install:
	# Locally install mapillary - DO THIS ONLY AFTER RUNNING `make build`
	@ pipenv run pip3 install -e .

## Build the package
build:
	# @ python3 -m build
	@ python3 setup.py sdist bdist_wheel --universal

## Clean directory of previous build
clean:
	@ rm -rf dist/ build/ src/*.egg-info

# TESTING

## Execute pytest on tests/
test:
	@ pytest --log-cli-level=20

test-no-warn:
	@ pytest --log-cli-level=20 --disable-warnings
