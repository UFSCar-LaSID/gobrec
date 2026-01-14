
.. image:: _static/GOBRecLogo.svg
   :align: center
   :width: 50%

GOBRec: GPU Optimized Bandits Recommender
=========================================

GOBRec is a Python library with an optimized implementation of contextual multi-armed bandits (CMABs) for recommender systems. The library has a simple API that allows you to use the CMAB algorithms to generate item (arms) expectations, using it for tasks other than recommendations. You can also use any of the implemented CMABs inside the Recommender to efficiently generate top-K recommendations.

The main contribution of GOBRec is its efficient implementation. With the vectorized code, using only CPU, our implementation was up to **150** times faster than other libraries. Using GPU optimization, our library achieved a speed gain of **700** times. More details about these comparisons can be found in the `benchmark page <benchmark.html>`_.

The GOBRec source code is available at: `https://github.com/UFSCar-LaSID/gobrec <https://github.com/UFSCar-LaSID/gobrec>`_

Documentation pages
===================

.. toctree::
   :maxdepth: 1

   installation
   quick start
   benchmark
   contributing
   public API