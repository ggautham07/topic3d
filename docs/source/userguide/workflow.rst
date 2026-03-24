Workflow
=====================

The ``topic3d runall`` workflow provides an end-to-end pipeline for topic
decomposition of single-cell chromatin tracing proximity data.

It is designed to take raw or preprocessed proximity datasets, convert them
into a representation suitable for topic modelling, train a Latent Dirichlet
Allocation (LDA) model, and save the resulting model and decomposition outputs.

Overview
--------

The pipeline performs the following steps:

1. Parse command-line arguments
2. Load training and optional test datasets
3. Prepare proximity data for topic modelling
4. Initialize the output directory structure and run metadata
5. Train an LDA model
6. Optionally evaluate learning using perplexity over iterations
7. Decompose the data into chromatin folding motifs (CFMs)
8. Save the trained model and decomposition results

Conceptual framework
--------------------

In ``topic3d``, chromatin tracing data is analysed using a topic modelling
framework inspired by natural language processing:

- **Single cells** are treated as documents
- **Pairwise proximities** are treated as words
- **Topics** are interpreted as chromatin folding motifs (CFMs)

This representation makes it possible to identify recurrent structural
patterns across many single cells in an unsupervised manner.

Pipeline structure
------------------

The ``runall`` workflow is built around the following internal steps.

1. Parse user inputs
^^^^^^^^^^^^^^^^^^^^

The workflow begins by parsing command-line arguments using the internal parser.
These arguments specify:

- input training data
- optional test data
- output directory
- model hyperparameters
- preprocessing parameters
- evaluation settings

If a JSON parameter file is provided, its contents are loaded and used to
supplement or replace command-line arguments.

2. Load input datasets
^^^^^^^^^^^^^^^^^^^^^^

The training dataset is loaded first. An optional test dataset may also be
loaded if provided.

The workflow expects the input files to contain chromatin tracing proximity
data in a format supported by the internal data-loading utilities.

At this stage, the raw file contents are read into memory but are not yet in
their final representation for topic modelling.

3. Prepare proximity data
^^^^^^^^^^^^^^^^^^^^^^^^^

The loaded data are converted into a topic-modelling-compatible representation
using the preprocessing utilities.

This step applies any supplied proximity-related parameters, such as threshold
or hardness settings, and transforms the input into a form suitable for LDA.

Typical preprocessing operations may include:

- interpreting pairwise contacts from the input data
- applying a proximity threshold
- converting continuous proximity values into weighted or binary features
- generating proximity vectors or matrices usable by the model

If both training and test datasets are provided, the same preprocessing logic
is applied to both.

4. Initialize output structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before model training begins, the workflow initializes the run environment.

This step creates and records:

- the output directory structure
- training-specific result paths
- run metadata
- configuration summaries
- optional identifiers for tracking the run

The initialization output is stored in a summary dictionary that is passed to
downstream steps of the workflow.

5. Train the LDA model
^^^^^^^^^^^^^^^^^^^^^^

The preprocessed training data are used to train a Latent Dirichlet Allocation
model.

Training behaviour depends on the supplied parameters, including:

- number of CFMs (topics)
- maximum number of iterations
- learning method (``batch`` or ``online``)
- batch size
- random seed
- whether to evaluate perplexity during training

An optional test dataset may be supplied for evaluation purposes.

If learning evaluation is enabled, the workflow records perplexity across
iterations and generates a plot showing how the model evolves during training.

6. Evaluate learning behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``evaluate_learning`` is enabled, the workflow stores the perplexity
trajectory returned during model training.

This information is then used to generate a figure showing the evolution of
perplexity over training iterations. This can be useful for:

- checking convergence
- diagnosing unstable training
- comparing training settings
- identifying diminishing returns from additional iterations

The resulting plot is saved in the training output directory.

7. Decompose the data
^^^^^^^^^^^^^^^^^^^^^

After training, the fitted model is used to decompose the processed training
data.

This step produces two main outputs:

``components``
    The learned topic-by-feature representation, corresponding to the CFMs
    inferred by the model.

``decomposition``
    The per-cell topic representation, describing how strongly each single cell
    is associated with each CFM.

These outputs form the basis for downstream interpretation and visualization.

8. Save results
^^^^^^^^^^^^^^^

Finally, the workflow saves the trained model and decomposition results to the
training output directory.

Depending on the internal save utilities, outputs may include:

- the trained LDA model
- decomposition matrices
- evaluation figures
- logs or summary files
- parameter records for reproducibility

Input data
----------

The workflow requires a training dataset and optionally accepts a separate test
dataset.

Training data
^^^^^^^^^^^^^

The training dataset is provided through ``--data_train_file`` and is required.

This dataset is used to:

- fit the topic model
- estimate chromatin folding motifs
- compute the decomposition of cells into topics

Test data
^^^^^^^^^

An optional test dataset can be provided through ``--data_test_file``.

If present, it may be used during model evaluation, for example when computing
perplexity or assessing generalization.

Preprocessing parameters
------------------------

The workflow extracts all parameters prefixed with ``proximity_`` and passes
them to the proximity-data preparation step.

Examples include:

- ``proximity_threshold``
- ``proximity_hardness``

These parameters control how raw proximity information is interpreted before
model training.

Model parameters
----------------

The most important modelling parameters include:

``num_CFMs``
    Number of chromatin folding motifs (topics) to infer.

``max_iter``
    Maximum number of training iterations.

``learning_method``
    Learning strategy for LDA (typically ``batch`` or ``online``).

``batch_size``
    Batch size for online learning.

``random_state``
    Random seed for reproducible training.

``evaluate_learning``
    Whether to record perplexity during learning.

``assess_model_quality``
    Whether to compute additional quality metrics after training.

Outputs
-------

The workflow produces a set of files in the results directory associated with
the training run.

These outputs typically include:

- a trained topic model
- decomposition results for the training data
- optional perplexity-evolution plots
- run metadata and configuration summaries

The exact file names and formats depend on the internal saving functions used
by the workflow.

Typical usage
-------------

A typical invocation of the full workflow looks like this:

.. code-block:: bash

   topic3d runall \
      --data_train_file ./data/train.npy \
      --data_test_file ./data/test.npy \
      --results_dir ./results/run_01 \
      --num_CFMs 15 \
      --max_iter 500 \
      --learning_method batch \
      --random_state 64 \
      --evaluate_learning

In this example, ``topic3d``:

- loads the training and test datasets
- prepares proximity features
- trains an LDA model with 15 CFMs
- tracks perplexity during learning
- decomposes the training data
- saves the trained model and outputs to ``./results/run_01``

Practical recommendations
-------------------------

For most users, the following guidelines are helpful:

- Start with a moderate number of CFMs, such as **5 to 20**
- Use ``batch`` learning unless the dataset is very large
- Set ``random_state`` for reproducibility
- Use ``evaluate_learning`` when testing convergence behaviour
- Compare several values of ``num_CFMs`` before selecting a final model

Interpreting the outputs
------------------------

The central outputs of the workflow are the learned CFMs and the per-cell
decomposition.

These can be interpreted as follows:

- **CFMs** represent recurrent chromatin proximity patterns across cells
- **Decomposition weights** indicate how strongly each cell expresses each CFM

These outputs can then be used for downstream analyses such as:

- comparing biological conditions
- identifying structural subpopulations
- visualizing topic embeddings
- relating structural states to perturbations or functional annotations

Notes
-----

The ``runall`` workflow is intended as a high-level entry point for users who
want to go from input data to trained model and decomposition results in a
single command.

More advanced users may prefer to call lower-level functions from
``topic3d.workflow`` or ``topic3d`` directly in order to customize individual
steps of preprocessing, training, decomposition, or post-processing.