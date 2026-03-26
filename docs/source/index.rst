.. toctree::
   :maxdepth: 4
   :caption: Quick Start
   :hidden:

   quickstart/install
   quickstart/run
   quickstart/tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   userguide/problem
   userguide/approach
   userguide/workflow
   userguide/interpretation

.. toctree::
   :maxdepth: 3
   :caption: Reference
   :hidden:

   reference/routines
   reference/parameters
   reference/api/modules


``topic3d`` documentation
==========================================

.. Current version: | release |
.. Release date: | today | 

``topic3d`` is a software package developed by the `Nollmann Lab <http://www.nollmannlab.org>`_ at the 
`Center for Structural Biology <http://www.cbs.cnrs.fr>`_, a research unit of the `CNRS <http://www.cnrs.fr>`_ 
and the `INSERM <http://www.inserm.fr>`_.


About
------------

``topic3d`` provides a robust pipeline and flexible API for modeling single-cell chromatin tracing data using Latent Dirichlet Allocation (LDA).

It treats single-trace proximity matrices as "documents" and pixels within as "words", and uses LDA to identify and quantify recurring "words" across "documents" called "topics", i.e., proximity patterns called **Chromatin Folding Motifs (CFMs)** across single traces.

More information is available in the User Guide and our methods `publication <https://www.biorxiv.org/content/10.1101/2024.09.18.613689v1>`_.


Quick start
------------

.. _cards-clickable:

.. grid:: 3
   :gutter: 2

   .. grid-item-card::
      Installing the package
      :img-bottom: _static/download-icon.png
      :img-alt:
      :link: quickstart/install
      :link-type: doc

   .. grid-item-card::
      Step-by-step walkthrough
      :img-bottom: _static/stairs-icon.png
      :img-alt:
      :link: quickstart/tutorials/pipeline
      :link-type: doc

   .. grid-item-card::
      Running the pipeline
      :img-bottom: _static/running-icon.png
      :img-alt:
      :link: quickstart/run
      :link-type: doc


Citation
------------

If you use topic3d in your research, please cite:

Gautham Ganesh, Jean-Bernard Fiche, Julien Mozziconacci, and Marcelo Nollmann, 2026 (*in preparation*)
**topic3d: an open-source software package to decompose single-cell pairwise proximity data using Latent Dirichlet
Allocation**