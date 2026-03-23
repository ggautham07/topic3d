Topic Modelling Parameters
=========================

This page describes the main parameters used in ``topic3d`` for training
topic models on chromatin tracing proximity datasets.

The parameters can be provided either:

- via the command-line interface (CLI)
- via a JSON parameters file using ``--params_file``

If a parameters file is provided, all command-line arguments are ignored.

---

Input / Output Parameters
------------------------

``--data_train_file`` : str (required)
    Path to the training dataset (NumPy ``.npy`` file).
    Each entry should represent a single-cell proximity matrix.

``--data_test_file`` : str (optional)
    Path to a test dataset used for model evaluation.

``--model_file`` : str (optional)
    Path to an existing trained model (``.joblib`` format).
    Can be used to resume or reuse a model.

``--results_dir`` : str (required)
    Directory where all outputs (model, metrics, logs) will be stored.

``--params_file`` : str (optional)
    Path to a JSON file containing all parameters.
    If provided, CLI parameters are ignored.

---

Data Preprocessing Parameters
-----------------------------

``--proximity_threshold`` : int (optional)
    Threshold applied to proximity values when working with
    FOF (frequency-of-fluorescence) data.

    Used to define whether two loci are considered "in proximity".

``--proximity_hardness`` : float, default=1
    Controls how strictly the proximity threshold is applied.

    - ``1`` → hard threshold (binary contacts)
    - ``<1`` → softer threshold (continuous weighting)

---

Core Model Parameters
---------------------

``--num_CFMs`` : int (required, range: 2–100)
    Number of **chromatin folding motifs (CFMs)**, i.e., topics.

    This is the most important hyperparameter.

    - Too small → merges distinct conformations (underfitting)
    - Too large → captures noise (overfitting)

    See the tutorial on selecting the number of CFMs for guidance.

``--max_iter`` : int, default=500
    Maximum number of training iterations.

    Increasing this improves convergence but increases runtime.

``--random_state`` : int, optional
    Random seed for reproducibility.

    Useful when comparing multiple runs or performing hyperparameter sweeps.

---

Learning Parameters
-------------------

``--learning_method`` : {``batch``, ``online``}, default=``batch``
    Training strategy for the LDA model.

    - ``batch`` → uses the full dataset at each iteration (more stable)
    - ``online`` → processes data in mini-batches (faster for large datasets)

``--batch_size`` : int, default=128
    Number of samples per batch when using ``online`` learning.

    Ignored when using ``batch`` mode.

---

Evaluation Parameters
---------------------

``--evaluate_learning`` : flag
    If enabled, computes model perplexity during training.

    - Provides insight into convergence
    - Increases training time

``--assess_model_quality`` : flag
    If enabled, computes additional quality metrics after training:

    - Perplexity
    - NPMI coherence score

    These metrics help evaluate the interpretability of CFMs.

---

Practical Recommendations
-------------------------

- Start with ``num_CFMs`` between **5–20** for most datasets
- Use ``batch`` learning unless working with very large datasets
- Set ``random_state`` for reproducibility
- Enable ``assess_model_quality`` when comparing models
- Use a parameter sweep to determine the optimal number of CFMs

---

Conceptual Reminder
-------------------

In ``topic3d``:

- Cells → documents  
- Pairwise proximities → words  
- CFMs → latent chromatin conformations  

The parameters above control how these latent structures are inferred
from single-cell data.