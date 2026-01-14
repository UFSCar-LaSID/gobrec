# Installation

There are two available ways to install the GOBRec library:

1. Installing from PyPI `pip install gobrec` (recommended);
2. Building from source code.

## Install from PyPI

GOBRec is available on [PyPI](https://pypi.org/project/gobrec/) and can be installed by the command below:

```bash
pip install gobrec
```

The recommended Python version to use is 3.8.20 (but newer versions should work too). For using GPU optimization, it is important to install PyTorch with CUDA implementation. More details on installing PyTorch with CUDA can be found in the [PyTorch documentation](https://pytorch.org/get-started/locally/). The recommended PyTorch version to use is 2.4.1.

## Build from source

It is possible to build the GOBRec from the source code. To do that, follow the commands below:

```bash
git clone https://github.com/RecSys-UFSCar/gobrec.git  # Clone our repository
cd gobrec
pip install setuptools wheel                           # Install the tools for generating the build
python setup.py sdist bdist_wheel                      # Generates the build
pip install dist/gobrec-X.X.X-py3-none-any.whl         # Install it using pip (replace the X.X.X with the desired version)
```
