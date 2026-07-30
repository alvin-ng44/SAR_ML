# Evaluating the effectiveness of the Soft Segmented Randomisation (SSR) Method

This repository evaluates the effectiveness of the SSR method and provides a way to reproduce our self-implemented version of it. It is not affiliated with, and was not produced by, any of the authors of the SSR paper.

We also extend the tests done by the authors of the SSR paper. For example, we also investigate the effect of the SSR method on OOD detection, Target Only, Target and Shadow Only classification and k Nearest Neighbours (target similarity search). 

## Table of contents

### [`Scenario 1a`](./notebooks/SSR/S1/Scenario_1a/) 
- **Train:** SAMPLE Synthetic, elevation of 14°, 15° and 16°
- **Test:** SAMPLE Measured, elevation of 17°

Results include accuracy, target similarity search, UMAPs and TSNEs. 

### [`Scenario 1b`](./notebooks/SSR/S1/Scenario_1b/)
- **Train:** SAMPLE Synthetic, elevation of 14°, 15°, 16° and 17°
- **Test:** SAMPLE Measured, elevation of 14°, 15°, 16° and 17°

Results include accuracy, UMAPs and TSNEs. 

### [`Scenario 2`](./notebooks/SSR/S2/)
- **Train:** SSR w noise augmented SAMPLE Synthetic
- **Test:** SAMPLE Measured, with MSTAR Clutter selected for each SAMPLE Measured chip

Results include accuracy and the effect of different MSTAR clutter types on the trained model.

### [`Segmentation`](./notebooks/SSR/Segmentation/) 
Here we investigate how the model performs if we only train and test on the target/target + shadow of the chips. Results include accuracy, UMAPs and TSNEs.

### [`Out Of Distribution Tests`](./notebooks/SSR/OOD/)
Here we compare the no augmentation method and the SSR method and measure their effect on OOD metrics such as AUROC and AURC. 

### [`Target Similarity Search`](./notebooks/SSR/Extras/target_similarity_search/)
Here we compare the L2 and Cosine similarity distance metrics to see which gives us nearest neighbours of closer azimuth angles. 

### [`Monte Carlo Dropout`](./notebooks/SSR/Extras/mc_dropout/)
Here we implement a Monte Carlo Dropout ResNet18 model and compare its results to a basic ResNet18 model's.

### [`Shift Invariant Models`](./notebooks/SSR/Extras/shifted_data/)
Here we test a shift invariant model on circular shifted SAMPLE Measured dataset. 

### [`DSO Data`](./notebooks/SSR/Extras/dso_data/)
Here we test the shift invariant model on a DSO SAR dataset. 

## Reference

Soft Segmented Randomisation reference: 

M. Kim et al., "Soft Segmented Randomization: Enhancing Domain Generalization in SAR ATR for Synthetic-to-Measured," in IEEE Access, vol. 12, pp. 175801-175816, 2024, doi: [10.1109/ACCESS.2024.3504574](https://doi.org/10.1109/ACCESS.2024.3504574)

SAMPLE Dataset reference:

B. Lewis, T. Scarnati, E. Sudkamp, J. Nehrbass, S. Rosencrantz and E. Zelnio, "A SAR Dataset for ATR Development: The Synthetic and Measured Paired Labeled Experiment (SAMPLE)," Proc. SPIE 10987, Algorithms for Synthetic Aperture Radar Imagery XXVI, Art. no. 109870H, May 2019, doi: [10.1117/12.2523460](https://doi.org/10.1117/12.2523460). 