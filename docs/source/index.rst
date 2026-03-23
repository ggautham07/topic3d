.. topic3d documentation master file, created by
   sphinx-quickstart on Sat Feb 21 14:35:27 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

topic3d documentation
=====================

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

`topic3d` is a software package developed by the `Nollmann Lab <http://www.nollmannlab.org>`_ at the `Center for Structural Biology <http://www.cbs.cnrs.fr>`_, a department of the `CNRS <http://www.cnrs.fr>`_ and the `INSERM <http://www.inserm.fr>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   about
   installation
   workflow
   parameters
   getting_started/index

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


.. ===============
.. Getting started
.. ===============

.. .. panels::
..     :card: + intro-card text-center
..     :body: text-center p-2
..     :column: col-lg-4 col-md-4 col-sm-6 col-xs-12 p-2

..     ---
..     .. image:: getting_started/tutorials/notebooks/_static/Download-Icon.png
..        :height: 180

..     +++

..     .. link-button:: getting_started/quick_install
..             :type: ref
..             :text: Installation
..             :classes: btn-block btn-info stretched-link text-white

..     ---
..     .. image:: _static/spot2matrix.png
..        :height: 180

..     +++

..     .. link-button:: getting_started/tutorials/notebooks/full_pyHiM_run
..             :type: ref
..             :text: 3D pipeline step by step
..             :classes: btn-block btn-info stretched-link text-white

..     ---
..     .. image:: _static/index_getting_started.svg
..        :height: 180

..     +++

..     .. link-button:: getting_started/typical_run
..             :type: ref
..             :text: Running pyHiM
..             :classes: btn-block btn-info stretched-link text-white



.. .. panels::
..     :card: + intro-card text-center
..     :body: text-center p-2
..     :footer: text-center p-2
..     :column: col-lg-2 col-md-4 col-sm-6 col-xs-12 p-2

..     ---
..     .. image:: _static/getting_started/dim_space.gif
..        :height: 100


..     +++

..     .. link-button:: user_guide/modules/preprocessing/make_projections
..             :type: ref
..             :text: 2D or 3D Pipeline
..             :classes: stretched-link

..     ---
..     .. image:: _static/registration_zoom.svg
..        :height: 100

..     +++

..     .. link-button:: user_guide/modules/preprocessing/align_images
..             :type: ref
..             :text: Image registration
..             :classes: stretched-link

..     ---
..     .. image:: _static/segmentation_mask.png
..        :width: 100

..     +++

..     .. link-button:: user_guide/modules/identification/segment_masks_3d
..             :type: ref
..             :text: Segment nuclei
..             :classes: stretched-link


..     ---
..     .. image:: _static/localizations.svg
..        :width: 100

..     +++

..     .. link-button:: user_guide/modules/identification/segment_sources_3d
..             :type: ref
..             :text: Localize spots
..             :classes: stretched-link

..     ---
..     .. image:: _static/trace_zoom.png
..        :height: 90

..     +++

..     .. link-button:: user_guide/modules/building_traces/build_traces
..             :type: ref
..             :text: Chromatin traces
..             :classes: stretched-link

..     ---
..     .. image:: _static/matrix_example.png
..        :width: 100

..     +++

..     .. link-button:: user_guide/modules/building_traces/build_matrices
..             :type: ref
..             :text: Distance maps
..             :classes: stretched-link


Installation
------------

.. code-block:: bash

   pip install topic3d


Quick Example
-------------

.. code-block:: bash

   topic3d runall \
      --data_train_file ./data/sample_sim_train.npy \
      --data_test_file ./data/sample_sim_test.npy \
      --results_dir ./results/ \
      --num_components 5 \
      --max_iter 100 \
      --random_state 64


Citation
--------

If you use topic3d in your research, please cite:

Gautham Ganesh, Jean-Bernard Fiche, Marcelo Nollmann, 2026 (in preparation)