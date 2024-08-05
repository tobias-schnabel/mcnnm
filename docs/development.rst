Development
===========

This guide is for developers who want to contribute to the lightweight-mcnnm project.

Setting up the Development Environment
--------------------------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/lightweight-mcnnm.git
      cd lightweight-mcnnm

2. Install Poetry (if not already installed):

   Follow the instructions on the `official Poetry website <https://python-poetry.org/docs/#installation>`_.

3. Install project dependencies:

   .. code-block:: bash

      poetry install

4. Activate the virtual environment:

   .. code-block:: bash

      poetry shell

Running Tests
-------------

To run the tests:

.. code-block:: bash

   poetry run pytest

Code Style
----------

We use Black for code formatting. To format your code:

.. code-block:: bash

   poetry run black .

Pre-commit Hooks
----------------
This project uses pre-commit hooks to ensure code quality and consistency. Pre-commit hooks are scripts that run automatically every time you commit changes to your version control system. They help catch common issues before they get into the codebase. To set up:

.. code-block:: bash

    poetry add pre-commit
    poetry run pre-commit install
    poetry run pre-commit run --all-files



Contributing
------------

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Write tests for your changes.
4. Implement your feature or bug fix.
5. Run the tests to ensure they pass.
6. Commit your changes and push to your fork.
7. Submit a pull request.

Building Documentation
----------------------

To build the documentation:

.. code-block:: bash

   cd docs
   make html

The built documentation will be in the `docs/_build/html` directory.
