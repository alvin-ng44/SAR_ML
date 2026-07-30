# Monte Carlo Dropout 

## Experimental Setup

This experiment uses the Monte Carlo Dropout ResNet18 model. More details of the model's implementation can be found from Lemay, A., Hoebel, K., Bridge, C.P. et al. Improving the repeatability of deep learning models with Monte Carlo dropout. npj Digit. Med. 5, 174 (2022). doi: [s41746-022-00709-3](https://doi.org/10.1038/s41746-022-00709-3). 


## Results

### Accuracy (Acc)

- **Train:** SAMPLE Synthetic
- **Test:** SAMPLE Measured

| Augmentation Method | Forward Passes | Mini | Mean | Max | Notebook |
|----------------------|--------|--------|----------|-----|----------|
| No Aug              | 100  | 80.07 | 85.04 | 91.75 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| SSR w noise         | 100  | 88.33 | 90.99 | 92.79 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| No Aug              | 200  | 79.93 | 85.05 | 91.75 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| SSR w noise         | 200  | 88.18 | 90.91 | 92.94 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| No Aug              | 400  | 80.15 | 85.07 | 91.67 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| SSR w noise         | 400  | 88.10 | 90.96 | 92.79 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| No Aug              | 800  | 80.07 | 85.08 | 91.67 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| SSR w noise         | 800  | 87.96 | 90.93 | 92.71 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| No Aug              | 1000 | 80.07 | 85.12 | 91.67 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| SSR w noise         | 1000 | 88.03 | 90.95 | 92.86 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |

### Other Metrics

| Augmentation Method | AURC | MCE   | ECE   | Notebook |
|---------------------|------|-------|-------|----------|
| No Aug              | 0.0343 | 0.2023 | 0.0656 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |
| SSR w noise         | 0.0177 | 0.2933 | 0.0475 | [`mc_dropout.ipynb`](./mc_dropout.ipynb) |

AURC uses entropy as a stand-in for confidence. 