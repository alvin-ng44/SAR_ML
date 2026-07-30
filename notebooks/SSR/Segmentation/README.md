# Experimental Setup
These set of experiments detail training and testing the classifier on the Target and Shadow only (that is, SAR images without clutter). Here, we use a segmentation algorithm inspired by (and modified from) Zhao's paper. This is because Choi segmentation allows the shadow to be picked anywhere from the paper, even if it is very far away from the target. Examples of Choi segmentation can be found here: [`Choi/choi_for_synth.ipynb`](./Choi/choi_for_synth.ipynb). To address this limitation, we use Zhao segmentation instead. Examples of Zhao segmentation can be found here: [`Zhao/zhao_for_synth.ipynb`](./Zhao/zhao_for_synth.ipynb).

## Results

### Accuracy (Acc)

| Augmentation | Segmentation         | Fill  | Min    | Max     | Mean   | Std   | Notebook  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| No  Aug      | Target + Shadow      | Zero  | 78.96  | 84.91   | 81.30  | 1.92  | [`Zhao/wo_aug.ipynb`](./Zhao/wo_aug.ipynb) |
| Gaussian     | Target + Shadow      | Zero  | 79.26  | 84.01   | 81.75  | 1.51  | [`Zhao/gaussian.ipynb`](./Zhao/gaussian.ipynb) |
| SSR w Noise  | Target + Shadow      | Zero  | 79.55  | 85.13   | 82.01  | 1.85  | [`Zhao/ssr_w_noise.ipynb`](./Zhao/ssr_w_noise.ipynb) |
| No  Aug      | Target Only      | Zero  | 77.10  | 83.72   | 80.51  | 1.98  | [`Zhao/wo_aug.ipynb`](./Zhao/wo_aug.ipynb) |
| Gaussian     | Target Only      | Zero  | 75.99  | 81.04   | 78.46  | 1.43  | [`Zhao/gaussian.ipynb`](./Zhao/gaussian.ipynb) |
| SSR w Noise  | Target Only      | Zero  | 70.71  | 76.43   | 73.30  | 1.81  | [`Zhao/ssr_w_noise.ipynb`](./Zhao/ssr_w_noise.ipynb) |

Zhao segmentation reference:
Z. Zhao, X. Xue, I. Mariam and X. Zhou, "Integrating Target and Shadow Features for SAR Target Recognition," Sensors, vol. 23, no. 19, p. 8031, 2023, doi: [10.3390/s23198031](https://doi.org/10.3390/s23198031).