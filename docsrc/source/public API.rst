

Public API
==========

.. toctree::
   :maxdepth: 3
   :titlesonly:

   public_api/recommender
   public_api/mabs/index

Library design
--------------

.. image:: _static/GOBRecDesign.png
   :align: center

The library leverages vectorized operations and optional GPU acceleration to enable efficient training and inference in large-scale settings. The library is structured around two core components: *(i)* the **MAB algorithm** and *(ii)* the **Recommender**, explained further in detail. Together, these components support incremental learning and the generation of top-K recommendations in an online setting.

* **MAB Algorithm:** This module is responsible for incremental model updates and executing exploration strategies. It provides optimized implementations of widely used linear CMAB methods, including **LinUCB** [1], **LinTS** [2], and **LinGreedy** [3]. All supported linear algorithms share a common ridge regression formulation for parameter estimation, which is encapsulated in a reusable base implementation to promote extensibility. In addition, GOBRec provides a **MABAlgo** interface that specifies the required methods and parameters for implementing new bandit algorithms that can be integrated into the recommendation pipeline.

* **Recommender:** It is responsible for efficiently ranking the item scores produced by an MAB algorithm and generating a top-K list of recommended items. It also handles the exclusion of previously consumed items from the recommendation set, ensuring that only eligible items are considered. The recommender operates independently of the underlying bandit implementation and can therefore be used with any algorithm conforming to the **MABAlgo** interface, facilitating the integration of new methods within the GOBRec framework.

The usage pipeline consists of feeding the recommender with context vectors, observed decisions (i.e., consumed item identifiers), and rewards (i.e., ratings or implicit feedback). These interactions are then used to update the underlying CMAB model incrementally. At inference time, new contexts are passed to the recommender, which invokes the MAB algorithm to score candidate items, filters previously consumed items, and returns a top-K recommendation list.