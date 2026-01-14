# Benchmark

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
        <th rowspan="13">
          GOBRec
        </th>
        <th colspan="10">
          MovieLens-100k
        </th>
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
        <th colspan="10">
          MovieLens-1M
        </th>
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
        <th colspan="10">
          MovieLens-10M
        </th>
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

The conducted experiments can be reproduced using the [code available in the `experiments` folder of GOBRec repository](https://github.com/UFSCar-LaSID/gobrec/tree/main/experiments).