
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

To install the necessary python packages, it is recommended to create an python 3.8 virtual environment. Using anaconda, it is possible to do that with the following command:

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

After this all necessary packages will be installed. It is possible to execute experiments with toy datasets only with this installation, but if you want to execute experiments with public dataset (e.g. MovieLens), it is necessary to install them. The next subsection helps with the installation of the available public datasets.

### Public datasets

## Executing experiments

## Generating metrics (plots)

## Expected results
