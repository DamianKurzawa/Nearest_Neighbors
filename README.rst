.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/my_knn_project.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/my_knn_project
    .. image:: https://readthedocs.org/projects/my_knn_project/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://my_knn_project.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/my_knn_project/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/my_knn_project
    .. image:: https://img.shields.io/pypi/v/my_knn_project.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/my_knn_project/
    .. image:: https://img.shields.io/conda/vn/conda-forge/my_knn_project.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/my_knn_project
    .. image:: https://pepy.tech/badge/my_knn_project/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/my_knn_project
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/my_knn_project

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==============
my_knn_project
==============


    Implementation and Analysis of the k-NN Algorithm


This project focuses on creating a custom implementation of the k-Nearest Neighbors (k-NN) algorithm without using ready-made solutions from libraries such as scikit-learn. The goal was to understand the algorithm from the ground up, compare it with an existing implementation, and prepare a full set of tests validating correctness.

The project includes an implementation of the SimpleKNN class, which supports different neighbor search strategies (brute-force, kd-tree, ball-tree), multiple distance metrics, and configuration parameters that mimic the behavior of a real-world k-NN model.

The input dataset is processed, split into training and testing sets, and used to compare the performance of the custom model with scikit-learn's implementation. The project also contains unit tests and consistency tests across different search strategies.

Project Scope
This project covers the following:
1. Custom k-NN Implementation
•	A fully implemented SimpleKNN class,
•	Support for parameters such as:
o	n_neighbors,
o	weights (uniform/distance),
o	metric (e.g., euclidean, manhattan),
o	leaf_size,
o	algorithm (brute, kd_tree, ball_tree),
•	Implementation of predict and predict_proba,
•	Manual construction of kd-tree and ball-tree structures.
2. Comparison with scikit-learn
•	Parameter tuning using GridSearchCV on the reference model,
•	Comparison of predictions in terms of:
o	accuracy,
o	ROC curve and AUC score,
o	prediction time,
o	classification consistency.
3. Unit Tests
Located in the tests/ directory:
•	tests for core class methods (fit, predict, predict_proba),
•	tests for brute-force vs kd-tree vs ball-tree behavior,
•	tests verifying dataset integrity after preprocessing,
•	tests comparing results with scikit-learn,
•	performance and stress tests.
4. PyScaffold-Based Structure
The project was generated using PyScaffold, providing:
•	a standardized project layout,
•	configuration files,
•	pytest integration,
•	documentation structure in the docs/ directory.
________________________________________
Results and Conclusions
The custom k-NN implementation produces results very close to the scikit-learn version.
Small differences may occur due to:
•	simplified kd-tree and ball-tree implementations,
•	different tie-breaking approaches,
•	optimizations used internally by scikit-learn.
Overall, the project provides a deeper understanding of the k-NN algorithm, the computational cost of nearest-neighbor search, and the role of data structures in accelerating the algorithm.


How to Run the Project
1. Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate
2. Install dependencies:
   pip install -r requirements.txt
3. Run the main script:
   python src/my_knn_project/main_knn.py
4. To run tests:
   pytest -v


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
