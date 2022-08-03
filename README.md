# ACCL Python Bindings

This package provides Python bindings for the [Alveo Collective Communication Library (ACCL)](https://github.com/Xilinx/ACCL).

## Installation
PyACCL is built and uploaded to PyPI with:
```sh
python3 setup.py sdist
python3 -m twine upload dist/*
```

Install it as follows:
<details>
  <summary>From PyPI</summary>

  ```sh
  python3 -m pip install pyaccl
  ```
</details>
<details>
  <summary>Locally from source distribution</summary>

  ```sh
  python3 -m pip install dist/*.tar.gz
  ```
</details>
<details>
  <summary>Locally editable</summary>

  ```sh
  python3 -m pip install -e .
  ```
</details>

## Downloading Notebooks and Overlays

PyACCL provides a few Jupyter notebooks and associated FPGA overlays for Alveo boards. After installing the PyACCL package, you can download these with:
```sh
pynq get-notebooks --from-package pyaccl all
```
Pynq will automatically download the right overlay for the Alveo device(s) in your system. If you do not have an Alveo board, add `--ignore-overlays` to the above command. The notebooks and overlays will be downloaded to the folder where the command was executed, and a Jupyter notebook server can be started from there.

There are several notebooks available to get you started with PyACCL:
* Intros to (Py)ACCL [primitives](src/pyaccl/notebooks/primitives.ipynb) and [collectives](src/pyaccl/notebooks/collectives.ipynb)
* Short guides to using [compression](src/pyaccl/notebooks/compression.ipynb) and [communicators](src/pyaccl/notebooks/communicators.ipynb)
* Quick overview of [performance-related flags](src/pyaccl/notebooks/performance.ipynb)
