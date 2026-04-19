# CS 4320 - Homework 12 Part A

This homework uses a small PyTorch CNN on the **Fashion-MNIST** dataset from Kaggle and downloads it with `kagglehub`.

Kaggle dataset:
`https://www.kaggle.com/datasets/zalando-research/fashionmnist`

## What this script does

- loads Fashion-MNIST from Kaggle CSV files
- creates train / validation / test splits
- checks batch shapes and dtypes
- trains a small CNN with a manual PyTorch training loop
- logs training and validation loss / accuracy each epoch
- saves the best checkpoint using validation accuracy
- reloads the checkpoint and verifies the validation result matches
- evaluates the best checkpoint on the test set only at the end
- saves a loss plot, JSON run summary, and a short reflection paragraph

## Main dependencies

- `torch`
- `pandas`
- `numpy`
- `matplotlib`
- `kagglehub`

Install `kagglehub` if needed:

```powershell
pip install kagglehub
```

If Kaggle asks for authentication, use your Kaggle API token. The current script is written to use `kagglehub.dataset_download(...)` directly.

## Run

```powershell
python cs-4320-hw12-part-a/cs-4320-hw12-part-a.py
```

## Output files

The script writes results into:

`cs-4320-hw12-part-a/outputs/`

Files created:

- `hw12_part_a_loss_curve.png`
- `best_fashion_mnist_cnn.pt`
- `hw12_part_a_run_summary.json`
- `hw12_part_a_reflection.txt`

## Notes

- The model is intentionally small to match the assignment constraints.
- The script was simplified to keep the focus on the core deep learning workflow.
- The test set stays isolated until the final evaluation step.
