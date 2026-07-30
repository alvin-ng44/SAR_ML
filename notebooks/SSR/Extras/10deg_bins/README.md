# Vehicle-Azimuth binning

## Experimental Setup

We partition each vehicle class into 10° azimuth bins (e.g., 10°–19° is one bin, 20°–29°
is another), and treat each vehicle-azimuth bin as its own class. Since this yields a small
sample size per class, we also test if training on one big dataset helps, that is we create 7 independent duplicates of the dataset, where whether SSR is applied to a given sample is drawn independently for
each duplicate (e.g. sample A may be augmented in copy 1 but not in copy 3). We also vary the probability of applying SSR (0.5 vs 0.8).


## Results

- **Train:** SAMPLE Synthetic, elevation of 14°, 15° and 16°
- **Test:** SAMPLE Measured, elevation of 17°

### Accuracy

#### Vehicle-Azimuth level

| Augmentation Method | Dataset Size | SSR probability | Min   | Max   | Mean  | Std  | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Aug              |     1x       |       -         | 42.12 | 50.28 | 45.19 | 2.49 | [`normal_size/wo_aug.ipynb`](./normal_size/wo_aug.ipynb) |
| Monte Carlo, No Aug |     1x       |       -         | 43.60 | 50.28 | 45.97 | 2.05 | [`normal_size/wo_aug_mc.ipynb`](./normal_size/wo_aug_mc.ipynb) |
| SSR w Noise         |     1x       |      0.5        | 61.04 | 69.02 | 66.81 | 2.16 | [`normal_size/ssr_w_noise.ipynb`](./normal_size/ssr_w_noise.ipynb) |
| SSR w Noise         |     7x       |      0.5        | 59.93 | 64.94 | 62.75 | 1.83 | [`7times_size/ssr_w_noise_big.ipynb`](./7times_size/ssr_w_noise_big.ipynb) |
| SSR w Noise         |     7x       |      0.8        | 62.52 | 69.57 | 66.09 | 2.32 | [`7times_size/ssr_w_noise_big_ssr80.ipynb`](./7times_size/ssr_w_noise_big_ssr80.ipynb) |
| Monte Carlo, SSR w Noise | 1x      |      0.5        | 67.53 | 73.28 | 69.57 | 1.93 | [`normal_size/ssr_w_noise_50_mc.ipynb`](./normal_size/ssr_w_noise_50_mc.ipynb) |
| Monte Carlo, SSR w Noise | 1x      |      0.8        | 67.16 | 72.36 | 69.78 | 1.55 | [`normal_size/ssr_w_noise_80_mc.ipynb`](./normal_size/ssr_w_noise_80_mc.ipynb) |
| Monte Carlo, SSR w Noise | 7x      |      0.8        | 65.12 | 72.36 | 68.76 | 1.95 | [`7times_size/ssr_w_noise_big_ssr80_mc.ipynb`](./7times_size/ssr_w_noise_big_ssr80_mc.ipynb) |



#### Vehicle level 

| Augmentation Method | Dataset Size | SSR probability | Min   | Max   | Mean  | Std  | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No Aug              |     1x       |       -         | 69.39 | 78.85 | 72.47 | 3.33 | [`normal_size/wo_aug.ipynb`](./normal_size/wo_aug.ipynb) |
| Monte Carlo, No Aug |     1x       |       -         | 68.27 | 77.18 | 73.38 | 2.96 | [`normal_size/wo_aug_mc.ipynb`](./normal_size/wo_aug_mc.ipynb) |
| SSR w Noise         |     1x       |      0.5        | 87.57 | 92.95 | 90.24 | 1.86 | [`normal_size/ssr_w_noise.ipynb`](./normal_size/ssr_w_noise.ipynb) |
| SSR w Noise         |     7x       |      0.5        | 87.38 | 90.54 | 88.89 | 1.10 | [`7times_size/ssr_w_noise_big.ipynb`](./7times_size/ssr_w_noise_big.ipynb) |
| SSR w Noise         |     7x       |      0.8        | 88.50 | 92.58 | 90.95 | 1.16 | [`7times_size/ssr_w_noise_big_ssr80.ipynb`](./7times_size/ssr_w_noise_big_ssr80.ipynb) |
| Monte Carlo, SSR w Noise | 1x      |      0.5        | 89.05 | 92.58 | 90.93 | 0.99 | [`normal_size/ssr_w_noise_50_mc.ipynb`](./normal_size/ssr_w_noise_50_mc.ipynb) |
| Monte Carlo, SSR w Noise | 1x      |      0.8        | 89.42 | 92.39 | 91.39 | 1.07 | [`normal_size/ssr_w_noise_80_mc.ipynb`](./normal_size/ssr_w_noise_80_mc.ipynb) |
| Monte Carlo, SSR w Noise | 7x      |      0.8        | 89.24 | 92.95 | 91.13 | 1.10 | [`7times_size/ssr_w_noise_big_ssr80_mc.ipynb`](./7times_size/ssr_w_noise_big_ssr80_mc.ipynb) |



### Target Similarity Search

For each Measured (17°) test chip, the 5 nearest neighbours (using Cosine distance) are retrieved from the Synthetic (14°/15°/16°) dataset. A chip is counted as a match if at least one of its five neighbours falls within a given azimuth angle threshold (1°, 3°, 5°) of the chip's true azimuth. Reported values are the percentage of correctly classified test chips matched at each threshold.

| Augmentation Method | Dataset Size | SSR probability | ≤1°   | ≤3°   | ≤5°   | Notebook |
| --- | --- | --- | --- | --- | --- | --- |
| No Aug              |     1x       |       -         | 56.6% | 76.5% | 85.3% | [`normal_size/wo_aug.ipynb`](./normal_size/wo_aug.ipynb) |
| Monte Carlo, No Aug |     1x       |       -         | 58.7% | 77.4% | 86.0% | [`normal_size/wo_aug_mc.ipynb`](./normal_size/wo_aug_mc.ipynb) |
| SSR w Noise         |     1x       |       0.5       | 60.2% | 82.6% | 91.7% | [`normal_size/ssr_w_noise.ipynb`](./normal_size/ssr_w_noise.ipynb) |
| SSR w Noise         |     7x       |      0.5        | 54.2% | 76.0% | 87.9% | [`7times_size/ssr_w_noise_big.ipynb`](./7times_size/ssr_w_noise_big.ipynb) |
| SSR w Noise         |     7x       |      0.8        | 57.6% | 82.2% | 93.6% | [`7times_size/ssr_w_noise_big_ssr80.ipynb`](./7times_size/ssr_w_noise_big_ssr80.ipynb) |
| Monte Carlo, SSR w Noise | 1x      |      0.5        | 63.0% | 86.5% | 94.5% | [`normal_size/ssr_w_noise_50_mc.ipynb`](./normal_size/ssr_w_noise_50_mc.ipynb) |
| Monte Carlo, SSR w Noise | 1x      |      0.8        | 65.8% | 90.7% | 96.9% | [`normal_size/ssr_w_noise_80_mc.ipynb`](./normal_size/ssr_w_noise_80_mc.ipynb) |
| Monte Carlo, SSR w Noise | 7x      |      0.8        | 58.4% | 82.7% | 93.1% | [`7times_size/ssr_w_noise_big_ssr80_mc.ipynb`](./7times_size/ssr_w_noise_big_ssr80_mc.ipynb) |


### Notes

All Monte Carlo dropout models are run 1000 times during inference. 