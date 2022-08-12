Getting Started
===============

.. _installation:

Installation
------------

Using pip from PyPI:

.. code:: sh

     python3 -m pip install pyaccl

Locally from source distribution:

.. code:: sh

     python3 setup.py sdist && python3 -m pip install dist/*.tar.gz

Locally editable:

.. code:: sh

     python3 -m pip install -e .

Downloading Notebooks and Overlays
----------------------------------

PyACCL provides a few Jupyter notebooks and associated FPGA overlays for
Alveo boards. After installing the PyACCL package, you can download
these with:

.. code:: sh

   pynq get-notebooks --from-package pyaccl all

Pynq will automatically download the right overlay for the Alveo
device(s) in your system. If you do not have an Alveo board, add
``--ignore-overlays`` to the above command. The notebooks and overlays
will be downloaded to the folder where the command was executed, and a
Jupyter notebook server can be started from there. 
There are several notebooks available to get you started with PyACCL:

* Intros to (Py)ACCL `primitives </src/pyaccl/notebooks/primitives.ipynb>`__ and `collectives  </src/pyaccl/notebooks/collectives.ipynb>`__
* Short guides to using `compression </src/pyaccl/notebooks/compression.ipynb>`__ and `communicators </src/pyaccl/notebooks/communicators.ipynb>`__ 
* Quick overview of `performance-related flags </src/pyaccl/notebooks/performance.ipynb>`__

Running Tests
-------------

PyACCL includes tests for single ACCL instances and systems of ACCL
instances. The tests are designed to run against ACCL emulator/simulator
sessions as well as the single-FPGA ACCL test overlay, which connects 3
ACCL instances on a single Alveo board.

To run the tests, add the ACCL emulator executable ``cclo_emu`` to your
path and start your emulator/simulator session:

.. code:: sh

   pyaccl-emulate -n <NRANKS>

then run the following command from the pyaccl root folder:

.. code:: sh

   mpirun -np <NRANKS> python3 -m pytest --with-mpi

If your system has an Alveo board and ``NRANKS`` is less or equal to 3,
the test fixture will try and download the appropriate overlay for it,
otherwise it will skip hardware testcases. Similarly, the test fixture
will attempt to identify a valid emulator/simulator session with the
appropriate number of ranks, then run testcases against it. If you omit
``--with-mpi``, only single-instance tests will run, i.e. tests for
``copy()`` and ``combine()`` primitives.


