# Environment Setup

1. Create and activate a new conda environment:
```bash
   conda create -n myenv python=3.12
   conda activate myenv
```

2. Install RAPIDS, PyTorch, and JupyterLab. The command differs by system:

   **NSCC** (CUDA 12.2 driver — override required; login nodes have no GPU visible at solve time):
```bash
   CONDA_OVERRIDE_CUDA="12.2" conda install -c rapidsai -c conda-forge -c nvidia \
       rapids=26.04 'cuda-version>=12.2,<=12.9' \
       'pytorch=*=*cuda*' 'torchvision=*=*cuda*' jupyterlab
```

   **Local** (CUDA 13 driver — auto-detected, no override needed):
```bash
   conda install -c rapidsai -c conda-forge -c nvidia \
       rapids=26.04 cuda-version=13.1 \
       'pytorch=*=*cuda*' 'torchvision=*=*cuda*' jupyterlab
```

3. Install remaining packages:
```bash
   conda install -c conda-forge matplotlib pandas torchmetrics lightning simpleitk scikit-image tifffile joblib
```

A full copy of the environment used for this project can be found at [`environment.yml`](./environment.yml).