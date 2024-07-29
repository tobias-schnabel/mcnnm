Installation
============

Requirements
------------
lightweight-mcnnm is compatible with Python 3.9 or later and depends on JAX and NumPy.

Installing from PyPI
--------------------
The simplest way to install lightweight-mcnnm and its dependencies is from PyPI using pip:

.. code-block:: bash

   $ pip install lightweight-mcnnm

Upgrading
---------
To upgrade lightweight-mcnnm to the latest version, use:

.. code-block:: bash

   $ pip install -U lightweight-mcnnm

Installing for Development
--------------------------
If you want to contribute to lightweight-mcnnm or modify the source code, you can install it in development mode:

1. Ensure you have Poetry installed. If not, install it by following the instructions on the `official Poetry website <https://python-poetry.org/docs/#installation>`_.

2. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/lightweight-mcnnm.git
      cd lightweight-mcnnm

3. Install the project dependencies:

   .. code-block:: bash

      poetry install

   This command creates a virtual environment and installs all the necessary dependencies.

4. Activate the virtual environment:

   .. code-block:: bash

      poetry shell

Now you're ready to start developing!
