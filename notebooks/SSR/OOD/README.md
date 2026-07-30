# Out of Distribution (OOD)

## Experimental Setup
These following experiments will explore whether the ssr method is robust to OODs. We train on 9 of the 10 SAMPLE classes cyclically using synthetic data, and test on all 10 measured classes. The OOD class will be the remaining class that the model has not countered.

## Results 

- **Test:** SAMPLE Measured
- **Protocol:** Leave-one-class-out OOD detection

| Augmentation   | OOD Class | AUROC  | AURC   | Notebook |
|----------------|-----------|:------:|:------:|----------|
| No Aug         | 2s1       | 0.8059 | 0.1119 | [`OOD_2s1.ipynb`](./OOD_2s1.ipynb) |
| SSR w noise    | 2s1       | 0.8528 | 0.0431 | [`OOD_2s1.ipynb`](./OOD_2s1.ipynb) |
| No Aug         | bmp2      | 0.8307 | 0.0790 | [`OOD_bmp2.ipynb`](./OOD_bmp2.ipynb) |
| SSR w noise    | bmp2      | 0.8765 | 0.0468 | [`OOD_bmp2.ipynb`](./OOD_bmp2.ipynb) |
| No Aug         | btr70     | 0.7807 | 0.0653 | [`OOD_btr70.ipynb`](./OOD_btr70.ipynb) |
| SSR w noise    | btr70     | 0.9039 | 0.0366 | [`OOD_btr70.ipynb`](./OOD_btr70.ipynb) |
| No Aug         | m1        | 0.7807 | 0.0653 | [`OOD_m1.ipynb`](./OOD_m1.ipynb) |
| SSR w noise    | m1        | 0.8244 | 0.1190 | [`OOD_m1.ipynb`](./OOD_m1.ipynb) |
| No Aug         | m2        | 0.7871 | 0.1212 | [`OOD_m2.ipynb`](./OOD_m2.ipynb) |
| SSR w noise    | m2        | 0.8984 | 0.0302 | [`OOD_m2.ipynb`](./OOD_m2.ipynb) |
| No Aug         | m35       | 0.6451 | 0.0967 | [`OOD_m35.ipynb`](./OOD_m35.ipynb) |
| SSR w noise    | m35       | 0.8077 | 0.0445 | [`OOD_m35.ipynb`](./OOD_m35.ipynb) |
| No Aug         | m60       | 0.7809 | 0.1072 | [`OOD_m60.ipynb`](./OOD_m60.ipynb) |
| SSR w noise    | m60       | 0.7587 | 0.0715 | [`OOD_m60.ipynb`](./OOD_m60.ipynb) |
| No Aug         | m548      | 0.6379 | 0.1079 | [`OOD_m548.ipynb`](./OOD_m548.ipynb) |
| SSR w noise    | m548      | 0.6690 | 0.0644 | [`OOD_m548.ipynb`](./OOD_m548.ipynb) |
| No Aug         | t72       | 0.7238 | 0.1708 | [`OOD_t72.ipynb`](./OOD_t72.ipynb) |
| SSR w noise    | t72       | 0.7533 | 0.0461 | [`OOD_t72.ipynb`](./OOD_t72.ipynb) |
| No Aug         | zsu23     | 0.6204 | 0.1480 | [`OOD_zsu23.ipynb`](./OOD_zsu23.ipynb) |
| SSR w noise    | zsu23     | 0.8286 | 0.0481 | [`OOD_zsu23.ipynb`](./OOD_zsu23.ipynb) |

A higher AUROC score means that the model is more reliable.
A lower AURC score indicates that the model is more accurate, as it means that there is lower risk at higher coverage.