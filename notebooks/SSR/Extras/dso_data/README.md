# DSO Dataset

## Experimental Setup

Here we use the updated GitHub LPD model on the DSO Dataset. The updated LPD models can be found here at [`LPD_github_model`](../../py_files/LPD_github/). It was originally downloaded at [`LPD GitHub Link`](https://github.com/raymondyeh07/learnable_polyphase_sampling/tree/main).

Images of the DSO Dataset can be found at [`plot_images.ipynb`](./cropped_images.ipynb). Comparisions of the Synthetic and Measured DSO Dataset can be found at [`plot_images.ipynb`](./cropped_images.ipynb) too. Note that there are no clutter for the Synthetic DSO dataset.

For the DSO Dataset to properly work on the LPD Models, we had to crop it. The cropped examples can be found here at [`cropped_images.ipynb`](./cropped_images.ipynb).

To apply SSR on the DSO Synthetic dataset, we had to segment out the target and shadow (using given precomputed masks) and paste them on MSTAR Clutter. The new images and the gmm segmentation (for ssr method) can be found here at [`images_on_mstar_bk.ipynb`](./images_on_mstar_bk.ipynb). 

## Accuracy

### Full Images

- **Train:** Full images, DSO Synthetic, elevation angles of 14°, 15° and 16°, azimuth angles of 10° to 79°.
- **Test:** Full images, DSO Measured, elevation angles of 17°, azimuth angles of 10° to 79°.

| Augmentation Method  | Min   | Max   | Mean  | Std  | Notebook |
| --- | --- | --- | --- | --- | --- |
| No Aug               | 15.81 | 39.74 | 29.15 | 6.28 | [`full_image/wo_aug_git.ipynb`](./full_image/wo_aug_git.ipynb) |
| Gaussian             | 24.36 | 56.41 | 40.17 | 9.93 | [`full_image/gaussian_git.ipynb`](./full_image/gaussian_git.ipynb) |
| SSR w Noise          | 29.06 | 47.86 | 37.86 | 6.46 | [`full_image/ssr_w_noise_git.ipynb`](./full_image/ssr_w_noise_git.ipynb) | 

### Target Only 

Segmentation was done using precomputed Target masks that were given. 

- **Train:** Target Only, DSO Synthetic, elevation angles of 14°, 15° and 16°, azimuth angles of 10° to 79°.
- **Test:** Target Only, DSO Measured, elevation angles of 17°, azimuth angles of 10° to 79°.

| Augmentation Method  | Min   | Max   | Mean  | Std  | Notebook |
| --- | --- | --- | --- | --- | --- |
| No Aug               | 25.21 | 34.62 | 30.30 | 2.87 | [`target_only/wo_aug_git.ipynb`](./target_only/wo_aug_git.ipynb) |
| Gaussian             | 33.76 | 41.45 | 36.41 | 2.81 | [`target_only/gaussian_git.ipynb`](./target_only/gaussian_git.ipynb) |

## Reference

LPD reference:

R. A. Rojas-Gomez, T.-Y. Lim, A. G. Schwing, M. N. Do and R. A. Yeh, "Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks," in Advances in Neural Information Processing Systems (NeurIPS), New Orleans, LA, USA, 2022.
