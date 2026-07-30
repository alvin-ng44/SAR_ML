# Shifted Data

## Experimental Setup

Here, we shift the SAMPLE Measured dataset using circular shift. Some examples of the shifted SAMPLE Measured dataset can be found at [`basic_rn18_and_visualisation.ipynb`](./basic_rn18_and_visualisation.ipynb). After which, we test a few shift invariant models (LPD, TIPS) and non-shift invariant models (Basic ResNet18 models) on the new shifted SAMPLE Measured dataset. We will report the accuracy on the unshifted SAMPLE Measured dataset. Model performance on the shifted SAMPLE Measured dataset can be found at the relevant linked notebooks.

## Models

The researchers' LPD model can be found here at: [`LPD GitHub Link`](https://github.com/raymondyeh07/learnable_polyphase_sampling/tree/main).

For our purposes, we have edited the code to make it work with new torch versions. The updated code can be found here at: [`LPD_github`](../../py_files/LPD_github/).

Since the LPD models have no dropout in-built and uses NLLLoss, we built a variant that has dropout enabled and uses CrossEntropyLoss at [`LPD_dropout`](../../py_files/LPD/lpd_resnet18.py).

We also built an equivalent LPD model (uses NLLLoss and dropout disabled) as the researchers' to ensure that our custom LPD models (using CrossEntropyLoss and dropout enabled) works. We do not report the results for this model but it can be found at [`LPD_equivalent`](../../py_files/LPD/lpd_resnet18_match.py).

Researchers' TIPS model can be found here at: [`TIPS GitHub Link`](https://github.com/sourajitcs/tips/). We do not report TIPS results, as its accuracy on the shifted SAMPLE Measured dataset did not match its accuracy on the unshifted SAMPLE Measured dataset. This suggests TIPS may be worth tuning further for SAR classification use. For reference, the notebook can be found at [`Archived/TIPS/ssr_w_noise.ipynb`](./Archived/TIPS/ssr_w_noise.ipynb).

## Accuracy

### Full Image

- **Train:** Full images, SAMPLE Synthetic, elevation angles of 14°, 15° and 16°.
- **Test:** Full images, unshifted SAMPLE Measured, elevation angles of 17°. Results for the shifted SAMPLE Measured (17°) can be found at their relevant notebook links. 

#### Non-shift invariant models

| Augmentation Method  | Min   | Max   | Mean  | Std  | Notebook |
| --- | --- | --- | --- | --- | --- |
| No Aug               | 61.78 | 75.70 | 68.29 | 4.39 | [`basic_rn18_and_visualisation.ipynb`](./basic_rn18_and_visualisation.ipynb) | 
| SSR w noise          | 86.27 | 94.62 | 91.91 | 2.36 | [`basic_rn18_and_visualisation.ipynb`](./basic_rn18_and_visualisation.ipynb) |
| SSR w noise, circular pad | 66.60 | 79.41 | 73.77 | 4.12 | [`LPD/CrossEntropy/full_image/ssr_w_noise_circularpad.ipynb`](./LPD/CrossEntropy/full_image/ssr_w_noise_circularpad.ipynb) |

No Aug and SSR w noise models were previously trained in [`S1/Scenario_1a/wo_aug.ipynb`](../../S1/Scenario_1a/wo_aug.ipynb) and [`S1/Scenario_1a/ssr_w_noise.ipynb`](../../S1/Scenario_1a/ssr_w_noise.ipynb) respectively.


The results of the above three models dropped substantially when test on shifted SAMPLE Measured (17°). See linked notebooks for exact values.

#### Shift Invariant models (LPD)

| Augmentation Method  | Dropout | Logits        | Loss         | Min   | Max   | Mean  | Std   | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Aug               | Yes     | 8, 16, 32, 64 | CrossEntropy | 31.91 | 57.70 | 44.56 | 10.63 | [`LPD/CrossEntropy/full_image/wo_aug.ipynb`](./LPD/CrossEntropy/full_image/wo_aug.ipynb) |
| Real (14°, 15°, 16°) | Yes     | 8, 16, 32, 64 | CrossEntropy | 99.07 | 100.00 | 99.72 | 0.25 | [`LPD/CrossEntropy/full_image/real.ipynb`](./LPD/CrossEntropy/full_image/real.ipynb) |
| GitHub, No Aug       | No      | 8, 16, 32, 64 | NLLLoss      | 37.66 | 53.99 | 43.64 | 5.62  | [`LPD/git_model/full_image/wo_aug_git.ipynb`](./LPD/git_model/full_image/wo_aug_git.ipynb) |
| GitHub, Real (14°, 15°, 16°) | No     | 8, 16, 32, 64 | NLLLoss | 99.07 | 100.00 | 99.52 | 0.27 | [`LPD/git_model/full_image/real_git.ipynb`](./LPD/git_model/full_image/real_git.ipynb) |
| SSR w Noise          | Yes     | 8, 16, 32, 64 | CrossEntropy | 68.46 | 78.29 | 73.25 | 3.53  | [`LPD/CrossEntropy/full_image/ssr_w_noise.ipynb`](./LPD/CrossEntropy/full_image/ssr_w_noise.ipynb) |
| GitHub, SSR w Noise  | No      | 8, 16, 32, 64 | NLLLoss      | 68.46 | 76.81 | 72.80 | 3.16  | [`LPD/git_model/full_image/ssr_w_noise_git.ipynb`](./LPD/git_model/full_image/ssr_w_noise_git.ipynb) | 
| SSR w Noise          | Yes     | 4, 8, 16, 32  | CrossEntropy | 74.40 | 81.08 | 77.55 | 2.23  | [`LPD/CrossEntropy/full_image/ssr_w_noise_logits_4_8_16_32.ipynb`](./LPD/CrossEntropy/full_image/ssr_w_noise_logits_4_8_16_32.ipynb) |

For the models with Augmentation Method "Real (14°, 15°, 16°)", those models were trained on SAMPLE Measured (14°, 15°, 16°) dataset. 

### Target Only

For this section, target only segmentation was done using the modified Zhao segmentation. After that, zero fill was used to fill the image's background. 

- **Train:** Target Only, SAMPLE Synthetic, elevation angles of 14°, 15° and 16°.
- **Test:** Target Only, unshifted SAMPLE Measured, elevation angles of 17°. Results for the shifted SAMPLE Measured (17°) can be found at their relevant notebook links. 

#### Shift Invariant models (LPD)

| Augmentation Method  | Dropout | Logits        | Loss         | Min   | Max   | Mean  | Std   | Notebook |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No Aug               | Yes     | 8, 16, 32, 64 | CrossEntropy | 58.26 | 69.20 | 62.73 | 3.16  | [`LPD/CrossEntropy/target_only/wo_aug.ipynb`](./LPD/CrossEntropy/target_only/wo_aug.ipynb) |
| GitHub, No Aug       | No      | 8, 16, 32, 64 | NLLLoss      | 57.51 | 67.90 | 62.99 | 2.92  | [`LPD/git_model/target_only/wo_aug_git.ipynb`](./LPD/git_model/target_only/wo_aug_git.ipynb) |
| Gaussian             | Yes     | 8, 16, 32, 64 | CrossEntropy | 54.17 | 59.93 | 57.05 | 1.67  | [`LPD/CrossEntropy/target_only/gaussian.ipynb`](./LPD/CrossEntropy/target_only/gaussian.ipynb) |
| GitHub, Gaussian     | No      | 8, 16, 32, 64 | NLLLoss      | 55.84 | 63.27 | 59.33 | 2.03  | [`LPD/git_model/target_only/gaussian_git.ipynb`](./LPD/git_model/target_only/gaussian_git.ipynb) |


## References

LPD reference:

R. A. Rojas-Gomez, T.-Y. Lim, A. G. Schwing, M. N. Do and R. A. Yeh, "Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks," in Advances in Neural Information Processing Systems (NeurIPS), New Orleans, LA, USA, 2022.

TIPS reference:

S. Saha and T. Gokhale, "Improving Shift Invariance in Convolutional Neural Networks with Translation Invariant Polyphase Sampling," 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Tucson, AZ, USA, 2025, pp. 620-629, doi: [10.1109/WACV61041.2025.00070](https://doi.org/10.1109/WACV61041.2025.00070). 

