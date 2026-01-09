
<p align="center">
  <img src="figs/GOBRecLogo.svg" alt="GOBRec Logo">
</p>

# GOBRec: GPU Optimized Bandits Recommender

GOBRec is a Python library with an optimized implementation of contextual multi-armed bandits (CMABs) for recommender systems. The library has a simple API that allows you to use the CMAB algorithms to generate item (arms) expectations, using it for tasks other than recommendations. You can also use any of the implemented CMABs inside the Recommender to efficiently generate top-K recommendations.

The main contribution of GOBRec is its efficient implementation. With the vectorized code, using only CPU, our implementation was up to X times faster than other libraries. Using GPU optimization, our library achieved a speed gain of Y times. More details about these comparisons can be found in the ["performance comparison" section](#performance-comparison) or in our [paper]().

The GOBRec documentation is available at: [https://recsys-ufscar.github.io/gobrec](https://recsys-ufscar.github.io/gobrec/).

## Installation

GOBRec is available on [PyPI](https://pypi.org/project/gobrec/) and can be installed by the command below:

```
pip install gobrec
```

The recommended Python version to use is 3.8.20 (but newer versions should work too). For using GPU optimization, it is important to install PyTorch with CUDA implementation. More details on installing PyTorch with CUDA can be found in the [PyTorch documentation](https://pytorch.org/get-started/locally/). The recommended PyTorch version to use is 2.4.1.

More installation options can be found in the [documentation]().

## Usage

This section shows two examples of how to use GOBRec. You can also use the available [Jupyter notebook](notebooks/usage_tutorial.ipynb) to reproduce the examples and verify the generated output.

### Using an MAB Algorithm individually to generate arm scores

It is possible to generate item (arm) expectations by using an MAB Algorithm alone. That way it is possible to use these algorithms for others tasks than recommendation.

```python
import numpy as np
# Import LinUCB as an example, it could be also LinTS or LinGreedy
from gobrec.mabs.lin_mabs import LinUCB

# A batch of contexts for training
contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# Corresponding decisions (items) taken, it can be str or int
decisions = np.array(['a', 1, 2])
# Corresponding rewards (ratings) received                     
rewards = np.array([1, 0, 1])

# Initialize the bandit. A seed is set for reproducibility and GPU usage can be switched
bandit = LinUCB(seed=42, use_gpu=True)

# Fit the model with the training data
bandit.fit(contexts, decisions, rewards)

# Predict scores for each arm (item) given a batch of contexts
bandit.predict(np.array([[1, 1, 0], [0, 1, 1]]))
```

### Using an MAB Algorithm to generate recommendations

It is possible to use an MAB Algorithm with the Recommender class to efficiently generate top-K recommendations.

```python
import numpy as np
import gobrec
# Import LinUCB as an example, it could be also LinTS or LinGreedy
from gobrec.mabs.lin_mabs import LinUCB

# A batch of contexts for training.
contexts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# Corresponding decisions (items) taken, it can be str or int
decisions = np.array(['a', 1, 2])
# Corresponding rewards (ratings) received
rewards = np.array([1, 0, 1])

recommender = gobrec.Recommender(
    # The recommender can use any implementation following the MABAlgo interface
    mab_algo=LinUCB(seed=42, use_gpu=True),
    # Number of items to recommend
    top_k=2
)

# Fit the model with the training data
recommender.fit(contexts, decisions, rewards)

# Recommend top_k items given a batch of contexts
recommender.recommend(np.array([[1, 1, 0], [0, 1, 1]]))
```

## Performance comparison

| Algorithm      | Time (m)       | Opt. mab2rec | Opt. iRec | Opt. CPU |
|----------------|----------------|--------------|-----------|----------|
| **MovieLens 100k** |                |              |           |          |
| mab2rec        | 0.8 (± 0.00)   |            |         |        |
| iRec           | 10.0 (± 0.02)  |            |        |       |
| gobrec CPU     | 0.01 (± 0.00)  | 106.7×       | 1308.1×   |       |
| gobrec GPU     | 0.00 (± 0.00)  | 192.1×       | 2354.7×   | 1.8×     |
| **MovieLens 1M** |                |              |           |          |
| mab2rec        | 18.0 (± 0.15)  |           |       |        |
| iRec           | 10.0 (± 0.02)  |            |         |        |
| gobrec CPU     | 0.11 (± 0.00)  | 168.6×       | 93.4×     |        |
| gobrec GPU     | 0.06 (± 0.00)  | 322.4×       | 178.7×    | 1.9×     |
| **MovieLens 10M** |                |              |           |          |
| mab2rec        | 406.5 (± 0.35) |            |         |        |
| iRec           | 10.0 (± 0.02)  |            |         |        |
| gobrec CPU     | 2.05 (± 0.02)  | 198.1×       | 4.9×      |       |
| gobrec GPU     | 0.85 (± 0.00)  | 476.3×       | 11.7×     | 2.4×     |


## Available algorithms

Available linear CMABs:

* [Lin](/gobrec/mabs/lin_mabs/lin.py) (only exploitation)
* [LinUCB](/gobrec/mabs/lin_mabs/lin_ucb.py) [1]
* [LinTS](/gobrec/mabs/lin_mabs/lin_ts.py) [2]
* [LinGreedy](/gobrec/mabs/lin_mabs/lin_greedy.py) [3]

Available baselines:

* [Random](/gobrec/mabs/random_mab.py)

## Contributing

Details on how to contribute to the GOBRec development can be viewed in the [contributing documentation](/CONTRIBUTING.md).

## License

GOBRec is licensed under the [MIT License](/LICENSE).

## Citation

TODO: PUT CITATION BIBTEX

## References

[1] Lihong Li, Wei Chu, John Langford, and Robert E. Schapire. A contextual-bandit 
    approach to personalized news article recommendation. In Proceedings of the 19th 
    International Conference on World Wide Web, WWW'09, pages 661-670, New York, NY, 
    USA, 2010. Association for Computing Machinery. doi: 10.1145/1772690.1772758.

[2] Shipra Agrawal and Navin Goyal. Thompson sampling for contextual bandits with 
    linear payoffs. In Proceedings of the 30th International Conference on Machine 
    Learning, ICML'13, pages 1220-1228, New York, NY, USA, 2013. JMLR.org. doi: 
    10.48550/arXiv.1209.3352.

[3] John Langford and Tong Zhang. The epoch-greedy algorithm for contextual multi-armed
     bandits. In Proceedings of the 20th International Conference on Neural Information 
     Processing Systems, NIPS'07, pages 817-824, Red Hook, NY, USA, 2007. Curran 
     Associates Inc. doi: 10.5555/2981562.2981665.
