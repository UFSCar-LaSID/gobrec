
# Contributing to GOBRec

In this page, we describe how you can contribute to the development of the GOBRec library. Firstly, we explain when to use issues or pull requests, and then some development tools are explained.

## Issues

Please, [open an issue](https://github.com/UFSCar-LaSID/gobrec/issues) mainly for these reasons:

* Report a bug
* Make suggestions for ways to improve the code
* Doubts about the library
* New algorithm ideas

When opening an issue, try to be as detailed as possible, giving examples, prints, error codes, environment specifications, etc.

## Pull requests

Please, make a [pull request](https://github.com/UFSCar-LaSID/gobrec/pulls) when:

* Fixed a bug
* Fixed some documentation
* Implemented a new algorithm
* Improved some already existing code

When making a pull request, try to be as detailed as possible, giving examples, prints, error codes, environment specifications, etc.

## Development

### Compiling the documentation

The documentation website is generated using [Sphinx](https://www.sphinx-doc.org/en/master/) and [numpydoc](https://numpydoc.readthedocs.io/en/latest/install.html). To compile it, first install the dependencies with the command below:

```
pip install -r requirements/docs.txt
```

Then, you can execute the following commands to generate the documentation in the HTML format:

```
sphinx-apidoc -o docsrc ./gobrec
cd docsrc
make html
```

### Generating a new version of the Python package (for PyPI)

First, it is necessary to update the library version inside the [setup.py](/setup.py).

Then, you can execute the following command to generate the library builds:

```
python setup.py sdist bdist_wheel
```


And finally submit it to PyPI with he command below:

```
twine upload dist/*
```
