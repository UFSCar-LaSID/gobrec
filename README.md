
<p align="center">
  <img src="figs/GOBRecLogo.svg" alt="GOBRec Logo">
</p>

# GOBRec: GPU Optimized Bandits Recommender

GOBRec is a Python library with an optimized implementation of contextual multi-armed bandits (CMABs) for recommender systems. The library has a simple API that allows you to use the CMAB algorithms to generate item (arms) expectations, using it for tasks other than recommendations. You can also use any of the implemented CMABs inside the Recommender to efficiently generate top-K recommendations.

The main contribution of GOBRec is its efficient implementation. With the vectorized code, using only CPU, our implementation was up to X times faster than other libraries. Using GPU optimization, our library achieved a speed gain of Y times. More details about these comparisons can be found in the ["performance comparison" section](#performance-comparison). For more detailed information please visit the [GOBRec documentation](https://ufscar-lasid.github.io/gobrec/).

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

To evaluate the computational efficiency of GOBRec, we compared its execution time against [Mab2Rec](https://github.com/fidelity/mab2rec) and [iRec](https://github.com/irec-org/irec) recommendation libraries on three [MovieLens](https://grouplens.org/datasets/movielens/) datasets of increasing scale.

Experiments were conducted in an incremental offline setting. The first 50% of interactions were used to warm up the models, while the remaining data were divided into ten equally sized windows. In each window, recommendations were generated, and the underlying models were incrementally updated using the observed decisions. Each experiment was repeated five times, with the average elapsed execution time and the speed-up achieved by GOBRec reported in the Table bellow.

<div>
  <table>
    <thead>
      <tr>
        <th></th>
        <th></th>
        <th colspan="3">LinGreedy</th>
        <th colspan="3">LinUCB</th>
        <th colspan="3">LinTS</th>
      </tr>
      <tr>
        <th></th>
        <th></th>
        <th></th><th>Mab2Rec</th><th>iRec</th>
        <th></th><th>Mab2Rec</th><th>iRec</th>
        <th></th><th>Mab2Rec</th><th>iRec</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="13">
          GOBRec
        </td>
        <td colspan="10" align="center">
          MovieLens-100k
        </td>
      </tr>
      <tr>
        <td></td>
        <td>Time</td><td>0.8</td><td>0.5</td>
        <td>Time</td><td>1.1</td><td>0.9</td>
        <td>Time</td><td>1.9</td><td>1.4</td>
      </tr>
      <tr>
        <td>CPU</td>
        <td>0.01</td><td>106.7×</td><td>66.7×</td>
        <td>0.07</td><td>15.6×</td><td>13.5×</td>
        <td>0.07</td><td>29.0×</td><td>21.4×</td>
      </tr>
      <tr>
        <td>GPU</td>
        <td>0.00</td><td>192.1×</td><td>120.6×</td>
        <td>0.01</td><td>102.5×</td><td>88.5×</td>
        <td>0.00</td><td>379.1×</td><td>279.3×</td>
      </tr>
      <tr>
        <td colspan="10" align="center">
          MovieLens-1M
        </td>
      </tr>
      <tr>
        <td></td>
        <td>Time</td><td>18.0</td><td>15.7</td>
        <td>Time</td><td>23.9</td><td>19.2</td>
        <td>Time</td><td>41.4</td><td>33.5</td>
      </tr>
      <tr>
        <td>CPU</td>
        <td>0.11</td><td>168.6×</td><td>147.1×</td>
        <td>1.32</td><td>18.0×</td><td>14.5×</td>
        <td>1.26</td><td>32.8×</td><td>26.5×</td>
      </tr>
      <tr>
        <td>GPU</td>
        <td>0.06</td><td>322.4×</td><td>281.2×</td>
        <td>0.20</td><td>117.4×</td><td>94.3×</td>
        <td>0.07</td><td>576.6×</td><td>466.6×</td>
      </tr>
      <tr>
        <td colspan="10" align="center">
          MovieLens-10M
        </td>
      </tr>
      <tr>
        <td></td>
        <td>Time</td><td>406.5</td><td>332.6</td>
        <td>Time</td><td>526.1</td><td>441.4</td>
        <td>Time</td><td>941.3</td><td>780.8</td>
      </tr>
      <tr>
        <td>CPU</td>
        <td>2.05</td><td>198.1×</td><td>162.1×</td>
        <td>28.21</td><td>18.7×</td><td>15.7×</td>
        <td>27.70</td><td>34.0×</td><td>28.2×</td>
      </tr>
      <tr>
        <td>GPU</td>
        <td>0.85</td><td>476.3×</td><td>389.7×</td>
        <td>4.13</td><td>127.4×</td><td>106.9×</td>
        <td>1.21</td><td>778.9×</td><td>646.1×</td>
      </tr>
    </tbody>
  </table>
</div>

The results highlight the computational efficiency of GOBRec, particularly for the LinGreedy and LinTS models, for which the GPU-enabled implementation achieves speed-ups of more than 400× and 700×, respectively, compared to Mab2Rec. Similar trends are observed in comparisons with iRec, where GOBRec consistently outperforms the baseline library across all evaluated CMAB models and datasets.

Scalability analysis reveals that GOBRec maintains near-linear time complexity relative to interaction volume; a 100× increase in data resulted in only a 121× increase in execution time for LinTS. In contrast, baselines exhibited super-linear growth (up to 558×), demonstrating GOBRec’s suitability for production-scale interaction matrices. Results show that even in scenarios with limited GPU availability, the optimized CPU implementation of GOBRec can substantially outperform competing libraries, achieving speed-ups of more than 100× for the LinGreedy model in all MovieLens datasets.

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
