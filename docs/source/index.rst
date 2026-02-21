.. topic3d documentation master file, created by
   sphinx-quickstart on Sat Feb 21 14:35:27 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

topic3d documentation
=====================

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about
   installation
   usage
   workflow
   cli

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules


About
------------

topic3d provides tools for preprocessing chromatin tracing data,
performing LDA-based topic decomposition of proximity matrices,
and post-processing results for biological interpretation.

topic3d treats single-cell proximity matrices as "documents"
and pixels as "words", allowing the extraction of latent
chromatin conformational patterns using LDA.


Installation
------------

.. code-block:: bash

   pip install topic3d


Quick Example
-------------

.. code-block:: bash

   topic3d-decompose \
      --data_train_file ./data/sample_sim_train.npy \
      --data_test_file ./data/sample_sim_test.npy \
      --results_dir ./results/ \
      --num_components 5 \
      --max_iter 50 \
      --learning_method "batch" \
      --random_state 64


Citation
--------

If you use topic3d in your research, please cite:

Gautham Ganesh, Jean-Bernard Fiche, Marcelo Nollmann, 2026 (in preparation)