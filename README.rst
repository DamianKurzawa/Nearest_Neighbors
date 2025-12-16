==============
Nearest_Neighbors Project
==============

Implementation and Analysis of the Nearest Neighbors Algorithm
==============================================================

This project focuses on creating a custom implementation of the **Nearest Neighbors (NN)** algorithm without using ready-made solutions from libraries such as scikit-learn.  
The main goal is to understand how nearest neighbor search works internally, how different search strategies affect performance, and how distance metrics influence neighbor selection.

Unlike k-NN classification, this project implements **pure nearest neighbor search**, meaning that the algorithm only finds the closest points in the dataset without assigning class labels.

The project includes a full implementation of a ``SimpleNearestNeighbors`` class and a direct comparison with ``sklearn.neighbors.NearestNeighbors``. The implementation is validated using unit tests and visualizations.


Project Scope
=============

1. Custom Nearest Neighbors Implementation
------------------------------------------

The project includes:

- A fully implemented ``SimpleNearestNeighbors`` class,
- Support for parameters such as:
  - ``n_neighbors`` – number of nearest neighbors,
  - ``radius`` – distance threshold for radius-based search,
  - ``metric`` (``euclidean``, ``manhattan``, ``minkowski``),
  - ``p`` – Minkowski distance parameter,
  - ``leaf_size`` – tree leaf size,
  - ``algorithm`` (``brute``, ``kd_tree``, ``ball_tree``),
  - ``exclude_self`` – option to ignore the query point itself,
- Implementation of:
  - ``fit``,
  - ``kneighbors``,
  - ``radius_neighbors``,
- Manual construction of **KD-tree** and **Ball-tree** data structures.


2. Comparison with scikit-learn
-------------------------------

The custom implementation is compared with
``sklearn.neighbors.NearestNeighbors`` in terms of:

- returned distances,
- returned neighbor indices,
- consistency across different algorithms,
- behavior of ``kneighbors`` vs ``radius_neighbors``.

The comparison demonstrates that the custom implementation produces identical or very similar results to the reference implementation.


3. Unit Tests
-------------

Tests are located in the ``tests/`` directory and include:

- tests for core methods (``fit``, ``kneighbors``, ``radius_neighbors``),
- tests verifying correct handling of ``exclude_self``,
- tests checking output shapes and distance ordering,
- tests comparing results with scikit-learn,
- edge-case and consistency tests.

All tests are executed using ``pytest``.


4. PyScaffold-Based Structure
-----------------------------

The project was generated using **PyScaffold**, providing:

- a standardized project layout,
- configuration files (``setup.cfg``, ``pyproject.toml``),
- pytest and coverage integration,
- documentation structure in the ``docs/`` directory.


Results and Conclusions
=======================

The custom Nearest Neighbors implementation produces results that closely match those of scikit-learn.

Small differences may occur due to:

- simplified KD-tree and Ball-tree implementations,
- different traversal orders,
- lack of low-level optimizations present in scikit-learn.

Overall, the project provides a deeper understanding of:

- nearest neighbor search algorithms,
- distance metrics and their impact,
- tree-based acceleration methods,
- differences between k-NN classification and pure nearest neighbor search.


How to Run the Project
=====================

1. Create and activate a virtual environment
--------------------------------------------

.. code-block:: bash

   python -m venv venv
   venv\Scripts\activate


2. Install dependencies
-----------------------

.. code-block:: bash

   pip install -r requirements.txt


3. Run the main script
---------------------

.. code-block:: bash

   python -m my_knn_project.main_nn


4. Run tests
------------

.. code-block:: bash

   pytest -v


Documentation
-------------------------------

- SimpleNearestNeighbors(
    n_neighbors=5,
    radius=1.0,
    algorithm="brute",
    leaf_size=30,
    metric="minkowski",
    p=2,
    exclude_self=False)
- fit(X) - Stores the training dataset and builds the selected data structure (KD-tree or Ball-tree).
- kneighbors(X_query, n_neighbors, return_distance=True) - Finds the k closest points to each query sample. Returns: distances to neighbors, indices of neighbors in the training dataset.
- radius_neighbors(X_query, radius, return_distance=True) - Finds all points within a given radius. Unlike kneighbors, the number of returned neighbors: may vary per query, may be zero or very large.


Notes
=====

This project has been set up using **PyScaffold 4.6**.  
For details and usage information on PyScaffold see https://pyscaffold.org/.
