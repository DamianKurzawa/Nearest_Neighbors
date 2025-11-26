==============
my_knn_project
==============

Implementation and Analysis of the k-NN Algorithm
================================================

This project focuses on creating a custom implementation of the k-Nearest Neighbors (k-NN) algorithm without using ready-made solutions from libraries such as scikit-learn. The goal was to understand the algorithm from the ground up, compare it with an existing implementation, and prepare a full set of tests validating correctness.

The project includes an implementation of the SimpleKNN class, which supports different neighbor search strategies (brute-force, kd-tree, ball-tree), multiple distance metrics, and configuration parameters that mimic the behavior of a real-world k-NN model.

The input dataset is processed, split into training and testing sets, and used to compare the performance of the custom model with scikit-learn's implementation. The project also contains unit tests and consistency tests across different search strategies.


Project Scope
=============

1. Custom k-NN Implementation
-----------------------------

The project includes:

- A fully implemented ``SimpleKNN`` class,
- Support for parameters such as:
  - ``n_neighbors``,
  - ``weights`` (``uniform`` or ``distance``),
  - ``metric`` (e.g. euclidean, manhattan),
  - ``leaf_size``,
  - ``algorithm`` (``brute``, ``kd_tree``, ``ball_tree``),
- Implementation of ``predict`` and ``predict_proba``,
- Manual construction of kd-tree and ball-tree structures.


2. Comparison with scikit-learn
-------------------------------

- Parameter tuning using ``GridSearchCV`` on the reference model,
- Comparison of predictions in terms of:
  - accuracy,
  - ROC curve and AUC score,
  - prediction time,
  - classification consistency.


3. Unit Tests
-------------

Tests are located in the ``tests/`` directory:

- tests for core class methods (``fit``, ``predict``, ``predict_proba``),
- tests for brute-force vs kd-tree vs ball-tree behavior,
- tests verifying dataset integrity after preprocessing,
- tests comparing results with scikit-learn,
- performance and stress tests.


4. PyScaffold-Based Structure
-----------------------------

The project was generated using PyScaffold, providing:

- standardized project layout,
- configuration files,
- pytest integration,
- documentation structure in the ``docs/`` directory.


Results and Conclusions
=======================

The custom k-NN implementation produces results very close to the scikit-learn version.

Small differences may occur due to:

- simplified kd-tree and ball-tree implementations,
- different tie-breaking approaches,
- scikit-learn internal optimizations.

Overall, the project provides a deeper understanding of the k-NN algorithm, the computational cost of nearest-neighbor search, and the role of data structures in accelerating the algorithm.


.. _pyscaffold-notes:

Notes
=====

This project has been set up using PyScaffold 4.6.  
For details and usage information on PyScaffold see https://pyscaffold.org/.
