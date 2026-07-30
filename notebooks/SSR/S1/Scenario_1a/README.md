# Scenario 1a

## Experimental Setup
These notebooks reproduce the First Scenario from M. Kim et al., "Soft Segmented Randomization: Enhancing Domain Generalization in SAR ATR for Synthetic-to-Measured," in IEEE Access, vol. 12, pp. 175801-175816, 2024, doi: [10.1109/ACCESS.2024.3504574](https://doi.org/10.1109/ACCESS.2024.3504574).

## Results

### Accuracy (Acc)

- **Train:** SAMPLE Synthetic, elevation of 14°, 15° and 16°
- **Test:** SAMPLE Measured, elevation of 17°

| Augmentation Method | Minimum | Maximum | Mean      | Std  | Notebook |
|----------------------|--------|--------|----------|-----|----------|
| No Aug               | 61.78   | 75.70   | 68.29     | 4.39 | [`wo_aug.ipynb`](./wo_aug.ipynb) |
| Gaussian             | 64.56   | 85.16   | 75.81     | 6.12 | [`gaussian.ipynb`](./gaussian.ipynb) |
| SSR w/o noise        | 87.20   | 93.69   | 89.81     | 2.13 | [`ssr_wo_noise.ipynb`](./ssr_wo_noise.ipynb) |
| SSR w noise         | 86.27   | 94.62   | 91.91 | 2.36 | [`ssr_w_noise.ipynb`](./ssr_w_noise.ipynb) |
| Measured 14°, 15° and 16° | 99.81 | 100.00 | 99.98 | 0.06 | [`train_real_14_15_16_test_real_17.ipynb`](./train_real_14_15_16_test_real_17.ipynb) |

- **Train:** SAMPLE Synthetic, elevation of 14°, 15° and 16°
- **Test:** SAMPLE Measured, elevation of 14°, 15°, 16° and 17°

| Augmentation Method | Minimum | Maximum | Mean      | Std  | Notebook |
|----------------------|--------|--------|----------|-----|----------|
| No Aug              | 63.42   | 76.21   | 70.49     | 3.97 | [`wo_aug.ipynb`](./wo_aug.ipynb) |
| SSR w noise         | 85.06   | 90.33   | 88.62 | 1.60 | [`ssr_w_noise.ipynb`](./ssr_w_noise.ipynb) |

### Target Similarity Search (TSS)

For each Measured (17°) test chip, the 5 nearest neighbours (using Cosine distance) are retrieved from the Synthetic (14°/15°/16°) dataset. A chip is counted as a match if at least one of its five neighbours falls within a given azimuth angle threshold (1°, 3°, 5°) of the chip's true azimuth. Reported values are the percentage of correctly classified test chips matched at each threshold.

| Augmentation Method | ≤1°   | ≤3°   | ≤5° | Notebook |
|----------------------|--------|--------|----------|----------|
| No Aug               | 22.3% | 37.3% | 46.8% | [`wo_aug.ipynb`](./wo_aug.ipynb) |
| SSR w noise         | 25.3% | 42.9% | 53.0% | [`ssr_w_noise.ipynb`](./ssr_w_noise.ipynb) |
| Measured 14°, 15° and 16° | 59.0% | 77.9% | 84.6% | [`train_real_14_15_16_test_real_17.ipynb`](./train_real_14_15_16_test_real_17.ipynb) |


### Notes

*For reference, [`train_real_14_15_16_test_real_17.ipynb`](./train_real_14_15_16_test_real_17.ipynb) reports the real-to-real upper bound, where training data is SAMPLE Measured 14°, 15° and 16° and test data is SAMPLE Measured 17°.*

The notebook [`Archived/simulation_fidelity.ipynb`](./Archived/simulation_fidelity.ipynb) compares the No Aug and Measured 14°, 15° and 16°. Those results can be found in [`wo_aug.ipynb`](./wo_aug.ipynb) and [`train_real_14_15_16_test_real_17.ipynb`](./train_real_14_15_16_test_real_17.ipynb) respectively. However, it has extra results for TSS (top 1 neighbour, azimuth angle threshold of 5°, 10°, 15°). 