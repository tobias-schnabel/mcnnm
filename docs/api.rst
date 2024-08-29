API Reference
=============

This page provides detailed information about the main functions in lightweight-mcnnm.

Main Functions
--------------

estimate
^^^^^^^^
.. autofunction:: mcnnm.wrappers.estimate

complete_matrix
^^^^^^^^^^^^^^^
.. autofunction:: mcnnm.wrappers.complete_matrix

Classes
-------

MCNNMResults
^^^^^^^^^^^^
.. autoclass:: mcnnm.wrappers.MCNNMResults
   :members:
   :noindex:

Internal Functions
------------------

Functions in wrappers.py
^^^^^^^^^^^^^^^^^^^^^^^^^

compute_treatment_effect
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.wrappers.compute_treatment_effect

Functions in validation.py
^^^^^^^^^^^^^^^^^^^^^^^^^^

cross_validate
~~~~~~~~~~~~~~
.. autofunction:: mcnnm.validation.cross_validate

holdout_validate
~~~~~~~~~~~~~~~~~
.. autofunction:: mcnnm.validation.holdout_validate

final_fit
~~~~~~~~~

.. autofunction:: mcnnm.validation.final_fit

Functions in core.py
^^^^^^^^^^^^^^^^^^^^^

initialize_coefficients
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core.initialize_coefficients

initialize_matrices
~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core.initialize_matrices

compute_svd
~~~~~~~~~~~

.. autofunction:: mcnnm.core.compute_svd

svt
~~~

.. autofunction:: mcnnm.core.svt

update_unit_fe
~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core.update_unit_fe

update_time_fe
~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core.update_time_fe

update_beta
~~~~~~~~~~~

.. autofunction:: mcnnm.core.update_beta

compute_Y_hat
~~~~~~~~~~~~~

.. autofunction:: mcnnm.core.compute_Y_hat

compute_objective_value
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core.compute_objective_value

initialize_fixed_effects_and_H
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core.initialize_fixed_effects_and_H

update_H
~~~~~~~~

.. autofunction:: mcnnm.core.update_H

update_L
~~~~~~~~

.. autofunction:: mcnnm.core.update_L

fit
~~~

.. autofunction:: mcnnm.core.fit


Functions in core_utils.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^

is_positive_definite
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core_utils.is_positive_definite

mask_observed
~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core_utils.mask_observed

mask_unobserved
~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core_utils.mask_unobserved

frobenius_norm
~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core_utils.frobenius_norm

nuclear_norm
~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core_utils.nuclear_norm

element_wise_l1_norm
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core_utils.element_wise_l1_norm

normalize
~~~~~~~~~~~

.. autofunction:: mcnnm.core_utils.normalize

normalize_back
~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.core_utils.normalize_back


Functions in utils.py
^^^^^^^^^^^^^^^^^^^^^^

convert_inputs
~~~~~~~~~~~~~~

.. autofunction:: mcnnm.utils.convert_inputs

check_inputs
~~~~~~~~~~~~

.. autofunction:: mcnnm.utils.check_inputs

generate_data
~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.utils.generate_data

propose_lambda_values
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.utils.propose_lambda_values

generate_lambda_grid
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.utils.generate_lambda_grid

extract_shortest_path
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.utils.extract_shortest_path

generate_holdout_val_defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.utils.generate_holdout_val_defaults

validate_holdout_config
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mcnnm.utils.validate_holdout_config
