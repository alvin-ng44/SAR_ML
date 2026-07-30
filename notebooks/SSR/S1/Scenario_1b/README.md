# Scenario 1b
These notebooks extend the experiments from the First Scenario of M. Kim et al., "Soft Segmented Randomization: Enhancing Domain Generalization in SAR ATR for Synthetic-to-Measured," in IEEE Access, vol. 12, pp. 175801-175816, 2024, doi: [10.1109/ACCESS.2024.3504574](https://doi.org/10.1109/ACCESS.2024.3504574).
The difference from the First Scenario of Kim's is the Training and Test datasets are extended to include the full SAMPLE Synthetic and Measured datasets respectively.

## Results

### Accuracy

- **Train:** SAMPLE Synthetic, elevation of 14°, 15°, 16° and 17°
- **Test:** SAMPLE Measured, elevation of 14°, 15°, 16° and 17°

| Augmentation Method | Minimum | Maximum | Mean      | Std  | Notebook |
|----------------------|--------|--------|----------|-----|----------|
| No Aug               | 79.41   | 86.69   | 82.81     | 2.59 | [`wo_aug_full_data.ipynb`](./wo_aug_full_data.ipynb) |
| SSR w noise         | 87.51   | 92.79   | 90.68 | 1.53 | [`ssr_w_noise_full_data.ipynb`](./ssr_w_noise_full_data.ipynb) |

### Notes

[`ssr_w_noise_umap_visualisation.ipynb`](./ssr_w_noise_umap_visualisation.ipynb) contains the normalised (using cosine method) UMAP using the model trained with ssr w noise augmented SAMPLE synthetic dataset. It contains two umap plots: 
1. the SAMPLE measured dataset and 200 epochs of ssr w noise SAMPLE synthetic dataset;
2. the SAMPLE measured dataset, the unaugmented SAMPLE synthetic dataset and the 200 epochs of ssr w noise SAMPLE synthetic dataset.

[`Archived/cosine_tsne_and_umap.ipynb`](./Archived/cosine_tsne_and_umap.ipynb) contains unnormalised and normalised UMAP and TSNE plots. It shows that the normalised UMAP and TSNE plots are more disk/circular shaped while the unnormalised UMAP and TSNE are more elongated in shape.