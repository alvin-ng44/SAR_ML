# Target Similarity Search

## Experimental Setup

We compare L2 and Cosine distance as similarity metrics for nearest-neighbour retrieval.
For each correctly classified Measured test chip, we retrieve its top-1 and top-5 nearest Synthetic neighbours under each metric, and check whether at least one retrieved neighbour falls within a given azimuth angle threshold (5°, 10°, 15°) of the chip's true azimuth.
We also show qualitative examples of the top synthetic neighbours retrieved by each metric
for selected measured samples.

## Results

- **Train:** SSR w noise augmented SAMPLE Synthetic
- **Test:** SAMPLE Measured

| Distance Metric | Top-N | ≤5°       | ≤10°      | ≤15°      | Notebook |
|-------------------|:-----:|----------:|----------:|----------:|----------|
| Cosine            | 1     | 20.9% | 37.2% | 53.1% | [`L2_and_cosine.ipynb`](./L2_and_cosine.ipynb) |
| L2                | 1     | 19.1%     | 34.9%     | 50.3%     | [`L2_and_cosine.ipynb`](./L2_and_cosine.ipynb) |
| Cosine            | 5     | 52.1% | 77.7% | 85.2% | [`L2_and_cosine.ipynb`](./L2_and_cosine.ipynb) |
| L2                | 5     | 51.0%     | 69.6%     | 83.3%     | [`L2_and_cosine.ipynb`](./L2_and_cosine.ipynb) |

## Notes
We also tested the Mahalanobis distance metric in [`Archived\Mahalanobis.ipynb`](./Archived/Mahalanobis.ipynb) but found that the nearest neighbours it found were not close to the true azimuth. 