
<p align="center">
  <img src="../figs/GOBRecLogo.svg" alt="GOBRec Logo">
</p>

<p align="center">
  ⚙️ <a href="#installation">Installation</a> |
  🚀 <a href="#executing-experiments">Executing Experiments</a> |
  📈 <a href="#generating-metrics-plots">Generating Metrics (Plots)</a> |
  ✅ <a href="#expected-results">Expected Results</a>
</p>

# Benchmark experiments

This folder contains the necessary scripts for executing the [GOBRec](https://github.com/UFSCar-LaSID/gobrec), [mab2rec](https://github.com/fidelity/mab2rec), and [iRec](https://github.com/irec-org/irec) benchmarks. The following sections explain how to correctly run the scripts, describing how to install the required software and data, how to execute the benchmarks and metrics generation scripts, and finally, the expected results that should be reached.

## Installation

It is necessary to make some installations before executing the experiments. In the following two sections, it will be explained how to install the necessary Python packages and the public datasets (e.g., MovieLens) to benchmark on them.

### Python packages

To install the necessary Python packages, it is recommended to create a Python 3.8 virtual environment. Using Anaconda, it is possible to do that with the following command:

```
conda create -n gobrec_benchmark python=3.8
```

And then activate this environment with the command:

```
conda activate gobrec_benchmark
```

Finally, use the `requirements.txt` to install the packages. Inside this folder (`experiments`), execute the following command:

```
pip install -r requirements.txt
```

After this, all necessary packages will be installed. It is possible to execute experiments with toy datasets only with this installation, but if you want to execute experiments with a public dataset (e.g., MovieLens), it is necessary to install it. The next subsection helps with the installation of the available public datasets.

### Public datasets

At the moment, we have benchmarks on [MovieLens](https://grouplens.org/datasets/movielens/). The list below gives the datasets' download links and how to install them to execute the experiments:

* [MovieLens-100K (ml-100k)](https://grouplens.org/datasets/movielens/100k/): download it and unzip in the `experiments/datasets` folder. You can maintain only the `experiments/datasets/ml-100k/u.data` and `experiments/datasets/ml-100k/u.item` files.
* [MovieLens-1M (ml-1m)](https://grouplens.org/datasets/movielens/1m/): download it and unzip in the `experiments/datasets` folder. You can maintain only the `experiments/datasets/ml-1m/ratings.dat` and `experiments/datasets/ml-1m/movies.dat` files.
* [MovieLens-10M (ml-10m)](https://grouplens.org/datasets/movielens/10m/): download it and unzip in the `experiments/datasets` folder. You can maintain only the `experiments/datasets/ml-10m/ratings.dat` and `experiments/datasets/ml-10m/movies.dat` files.

## Executing experiments

After installing the necessary Python packages and datasets, it is possible to start executing the experiments. To do that, execute the following command:

```
python execution_time_test.py
```

Executing this Python code will ask you which datasets, algorithms, and experiment type to use. Input the datasets, algorithms, and experiment type indexes separated by space to select the wanted options.

Another way to select the options is by executing the command below:

```
python execution_time_test.py --datasets <datasets> --algorithms <algorithms> --experiments <experiments>
```

Replace `<datasets>` with the names (or indexes) of the datasets separated by commas (","). The available datasets to use are:

- \[1\]: toy_dataset_1k_100k
- \[2\]: toy_dataset_1k_500k
- \[3\]: toy_dataset_1k_1M
- \[4\]: toy_dataset_1k_5M
- \[5\]: toy_dataset_1k_15M
- \[6\]: toy_dataset_1k_30M
- \[7\]: toy_dataset_10k_100k
- \[8\]: toy_dataset_10k_500k
- \[9\]: toy_dataset_10k_1M
- \[10\]: toy_dataset_10k_5M
- \[11\]: toy_dataset_10k_15M
- \[12\]: toy_dataset_10k_30M
- \[13\]: toy_dataset_100k_500k
- \[14\]: toy_dataset_100k_1M
- \[15\]: toy_dataset_100k_5M
- \[16\]: toy_dataset_100k_15M
- \[17\]: toy_dataset_100k_30M
- \[18\]: ml-100k
- \[19\]: ml-1m
- \[20\]: ml-10m
- all (it will use all datasets)

Replace `<algorithms>` with the names (or indexes) of the algorithms separated by commas (","). The available algorithms to execute are:

- \[1\]: mab2rec_LinUCB
- \[2\]: mab2rec_LinGreedy
- \[3\]: mab2rec_LinTS
- \[4\]: gobrec_LinUCB_CPU
- \[5\]: gobrec_LinGreedy_CPU
- \[6\]: gobrec_LinTS_CPU
- \[7\]: gobrec_LinUCB_GPU
- \[8\]: gobrec_LinGreedy_GPU
- \[9\]: gobrec_LinTS_GPU
- \[10\]: irec_LinUCB
- \[11\]: irec_LinGreedy
- \[12\]: irec_LinTS
- all (it will use all algorithms)

Replace `<experiments>` with the names (or indexes) of the experiments separated by commas (","). The available experiments to execute are:

- \[1\]: not_incremental
- \[2\]: incremental
- all (it will use all experiment types)

The `not_incremental` experiment will train the algorithm on 50% of the dataset and then generate the recommendations for the other 50%. In the `incremental` setup, the algorithm will be trained on 50% of the dataset, and the rest of the dataset will be split into 10 windows, making recommendations and training incrementally on each window.

## Generating metrics (plots and tables)

After executing the experiments, you will have the necessary data to generate the plots and tables. You can create them by using the `generate_metrics.ipynb` Jupyter notebook. In the notebook cells, you can change the datasets, algorithms, and other settings of the plots and tables. The important codes that could be modified to change these configurations are commented with `CHANGE HERE`.

## Expected results

Originally, the experiments were conducted on a workstation equipped with an NVIDIA RTX 3090 GPU and an Intel Xeon E5-2630 v4 CPU, with and without GPU acceleration. If executed in the same architecture mentioned, it is expected to obtain the results shown in the table below:

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