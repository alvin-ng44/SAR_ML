# Scenario 2

## Experimental Setup
These notebooks reproduce the Second Scenario from M. Kim et al., "Soft Segmented Randomization: Enhancing Domain Generalization in SAR ATR for Synthetic-to-Measured," in IEEE Access, vol. 12, pp. 175801-175816, 2024, doi: [10.1109/ACCESS.2024.3504574](https://doi.org/10.1109/ACCESS.2024.3504574).

Instead of using the Choi segmentation used in Kim's paper, we use a modified version of Choi segmentation, which calibrates the intensities of both the target and shadow. 

## Results

### Accuracy (Acc)

- **Train:** SSR w noise augmented SAMPLE Synthetic
- **Test:** SAMPLE Measured, with MSTAR Clutter selected for each SAMPLE Measured chip

|   Clutter type    | Minimum | Maximum | Mean  | Std  | Notebook |
| --- | --- | --- | --- | --- | --- |
| Randomly Selected | 77.62   | 83.94   | 81.93 | 1.62 | [`ssr_w_noise.ipynb`](./ssr_w_noise.ipynb) |
|       Bright      | 80.15   | 89.96   | 84.96 | 2.61 | [`clipped_pixles.ipynb`](./clipped_pixels.ipynb) |
|       Dark        | 88.03   | 92.12   | 90.51 | 1.33 | [`clipped_pixles.ipynb`](./clipped_pixels.ipynb) |
|       Mixed       |  57.40  | 71.97   | 64.80 | 3.63 | [`clipped_pixles.ipynb`](./clipped_pixels.ipynb) |

The "Bright", "Dark" and "Mixed" MSTAR clutter can be found in [`clipped_pixles.ipynb`](./clipped_pixels.ipynb). 

Choi segmentation reference: 

J. -H. Choi, M. -J. Lee, N. -H. Jeong, G. Lee and K. -T. Kim, "Fusion of Target and Shadow Regions for Improved SAR ATR," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-17, 2022, Art no. 5226217, doi: [10.1109/TGRS.2022.3165849](https://doi.org/10.1109/TGRS.2022.3165849).