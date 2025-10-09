# PyTorch Project Blueprint

This repository contains a blueprint that I use as the foundation for all my PyTorch projects.  
The goal is to provide a minimal, easy‑to‑understand code base that remains efficient and flexible.

## Deploy

Clone the repository and run the `deploy.sh` script.  
The script will rename the project and delete itself; after it finishes you are ready to go.

## Dependencies

My projects rely on the following libraries:
- **fabric** – A flexible and efficient way to manage multi‑GPU settings.  
- **hydra** – Elegantly manages dependencies in a flexible way.  
- **matplotlib** – For visualizing results.  
- **pandas** – For processing data frames.  
- **pre‑commit** – Adds pre‑commit hooks for code formatting.  
- **pytest** – To run unit tests.  
- **scipy** – Contains convenient algorithms that are useful.  
- **setuptools** – For building the project.  
- **sigfig** – For accurate metric plotting.  
- **tensorboard** – For plotting training curves.
- **torch** – Basic machine‑learning framework with GPU support and autograd.  
- **torchmetrics** – Convenient metric wrappers (especially for multiple GPUs).  

## Project Structure

```
scripts/   – Entry‑point scripts (training, validation, inference, …)
data/      – Raw data used in datasets (currently empty)
outputs/   – Training runs, logs, and checkpoints
configs/   – Hydra configuration files
tests/     – Unit tests for critical parts of the project
src/       – Core of the project
```

Inside `src` you’ll find a classic architecture:

```
datasets/  – Custom dataset classes
engine/    – Training and validation loops
losses/    – Loss functions
metrics/   – Additional metrics
models/    – Model definitions
trainers/  – Training‑step logic
utils/     – Small, self‑contained helper functions
```

The `utils` directory should contain very minimal functions, each saved in a descriptively named file.  