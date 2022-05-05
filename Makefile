archive:
	python3 -m twine upload --repository testpypi dist/* --verbose

build:
	python3 -m build

clean:
	rm -rf dist/ src/*.egg-info