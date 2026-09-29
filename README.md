
# NNDL Saliency Project

Visual saliency prediction on **SALICON** using PyTorch.

This project develops and evaluates a controlled family of saliency prediction models, from a non-neural center-prior baseline to CNN- and Transformer-based architectures. Each experiment changes one main component at a time including feature extraction, multi-scale fusion, training objective, and adaptive spatial priors while keeping the data splits, preprocessing, and evaluation protocol consistent across models.

## Index

- [Model Variants and Comparisons](#model-variants-and-comparisons)
- [Repository Structure](#repository-structure)
- [Setup and Requirements](#setup-and-requirements)
- [Dataset and Data Splits](#dataset-and-data-splits)
- [Training and Evaluation Protocol](#training-and-evaluation-protocol)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Model Variants and Comparisons

The model family was designed to study the contribution of individual modeling choices while keeping the remaining components as comparable as possible. **B0** serves as an independent non-neural reference baseline, while the main neural progression is **B1 → M1 → M1-L**. From M1-L, two alternative extensions are evaluated: **G**, which introduces an adaptive center prior, and **M2**, which replaces the CNN encoder with a hierarchical Transformer.

| Model | Main change | Core configuration | Purpose |
| :--- | :--- | :--- | :--- |
| **B0** | Center-prior baseline | Mean saliency map computed from the training set; no backpropagation | Measure how much of the saliency distribution can be explained by center bias alone |
| **B1** | Neural baseline | Pretrained ResNet18 using the deepest feature map (C5) + lightweight decoder; MSE loss | Establish the neural reference model |
| **M1** | Multi-scale feature fusion | ResNet18 features from C3/C4/C5 + multi-scale skip connections; same MSE loss as B1 | Measure the contribution of multi-scale information |
| **M1-L** | Training objective | Same architecture as M1; `0.5 × (1 - CC) + 0.5 × KLD` | Measure the effect of replacing MSE with a saliency-specific objective |
| **G** | Adaptive center prior | Frozen M1-L + frozen B0 combined through a learned image-dependent gate | Test whether the contribution of center bias can be beneficial when adapted to each image |
| **M2** | Encoder replacement | Pretrained PVTv2-B1 encoder + the same multi-scale decoder design and CC+KLD objective as M1-L | Study the effect of replacing the CNN encoder with a hierarchical Transformer |

The experimental analysis is based on the following paired comparisons:

| Comparison | Question addressed |
| :--- | :--- |
| `B1 - B0` | How much does a learned neural predictor improve over a static center-prior baseline? |
| `M1 - B1` | What is the effect of introducing multi-scale feature fusion and skip connections? |
| `M1-L - M1` | What is the effect of replacing MSE with the CC+KLD training objective? |
| `G - M1-L` | Does an image-dependent adaptive center prior improve the M1-L prediction? |
| `M2 - M1-L` | What is the effect of replacing the ResNet18 encoder with PVTv2-B1 while keeping the decoder design and training objective comparable? |

This structure allows each comparison to be associated with a specific modeling decision, while B0 remains a reference baseline rather than a strict architectural ablation.

## Repository Structure

The repository is organized to keep configuration, data handling, model definitions, training, evaluation, and analysis clearly separated.

```text
requirements-colab.txt        # Extra dependencies for Google Colab
environment-mac.yml            # Conda environment for local macOS development

configs/
├── data.yaml                  # Shared dataset and preprocessing settings
└── experiments.yaml           # Model, loss, training, and optimization settings

notebooks/
└── colab_bootstrap.ipynb      # Main Colab workflow for setup, training, and evaluation

scripts/
├── audit_dataset.py           # Dataset validation
├── download_salicon.py        # SALICON download utilities
├── train.py                   # Model training
├── evaluate.py                # Model evaluation
├── compare_evaluations.py     # Per-image paired comparisons
├── bootstrap_ci.py            # Bootstrap confidence intervals
├── diagnose_gate.py           # Adaptive center-prior gate diagnostics
├── qualitative_figures.py     # Qualitative saliency visualizations
└── smoke_test.py              # End-to-end pipeline sanity check

scripts/docs/
└── evaluation_protocol.md     # Evaluation protocol and comparison rules

src/
├── data/                      # PyTorch dataset, fixed splits, fixation handling
├── losses/                    # Differentiable saliency losses
├── models/                    # Model architectures and model factory
├── checkpoints.py             # Checkpoint loading and validation
├── config_utils.py            # Configuration utilities
├── evaluation.py              # Shared evaluation logic
├── metrics.py                 # Saliency evaluation metrics
├── runtime.py                 # Device and runtime utilities
├── saliency_maps.py           # Saliency-map processing utilities
└── training_monitoring.py     # Training history and monitoring

tests/
└── ...                        # Unit and integration tests

results/
├── split_manifest.csv         # Fixed train/tuning/internal-test split
└── ...                        # Evaluation and comparison artifacts
```

## Setup and Requirements

The project can be run either on **Google Colab**, which is the recommended environment for training and evaluation, or locally on macOS for development and lightweight testing.

### Pretrained Weights

B1, M1, M1-L, and G use an ImageNet-pretrained ResNet18 (from `torchvision`), while M2 uses a pretrained PVTv2-B1 (from `timm`). Pretrained weights are downloaded automatically the first time training is run, so an internet connection is required on the first run.

### Google Colab

The recommended entry point is `notebooks/colab_bootstrap.ipynb`, which follows this sequence:

1. verify the GPU runtime
2. mount Google Drive
3. clone / update the repository
4. install project dependencies
5. prepare the local SALICON cache
6. run tests and smoke checks
7. training / evaluation / analysis

Install the required dependencies from the repository root with:

```bash
python -m pip install -r requirements-colab.txt
```

The main test suite and end-to-end smoke test can then be run with:

```bash
python -m pytest -q
python scripts/smoke_test.py
```

`requirements-colab.txt` is designed to **extend** the standard Google Colab environment (which already ships torch/torchvision/numpy/matplotlib with GPU support), rather than describe a completely empty Python installation.

### Local macOS Environment

For local development, create the Conda environment with:

```bash
conda env create -f environment-mac.yml
conda activate nndl-saliency
```

The local environment is mainly intended for code development, lightweight debugging, and automated tests, while GPU-intensive training is typically performed on Colab. At runtime, the project selects the best available backend in the following order:

**CUDA → Apple MPS → CPU**

## Dataset and Data Splits

The project uses the **SALICON** dataset. Dataset download and validation utilities are provided in:

```text
scripts/download_salicon.py
scripts/audit_dataset.py
```

### Setup Kaggle API

The dataset is hosted on Kaggle (`roshan401/salicon`), so `scripts/download_salicon.py` needs a **personal** Kaggle API token. Anyone reproducing the project should use their own free Kaggle account.

1. Create an API token from your Kaggle account settings (API section). This downloads a `kaggle.json` file.
2. Save it as `~/.kaggle/kaggle.json` (on Google Colab, upload it to `/root/.kaggle/kaggle.json`).
3. On Linux/macOS, restrict its permissions: `chmod 600 ~/.kaggle/kaggle.json`.

Never commit `kaggle.json` to the repository. If SALICON is already available locally, the download step can be skipped: pass its location with `--data_dir` to the other scripts, keeping the same folder layout (verify it with `scripts/audit_dataset.py`).

### Download and Validation

To download and verify the dataset:

```bash
python scripts/download_salicon.py --output_dir data/
python scripts/audit_dataset.py --data_dir data/
```

The project uses SALICON images and saliency maps. Fixation files are also retained because fixation-based metrics such as NSS and sAUC are implemented in the codebase.

When running on Google Colab, the dataset is copied or extracted to the local VM storage, typically:

```text
/content/data_local
```

This avoids the I/O overhead of reading thousands of images directly from mounted Google Drive.

### Data Splits

All experiments use the same fixed split defined in:

```text
results/split_manifest.csv
```

The manifest is shared across all experiments and must not be regenerated independently. `tuning` and `internal_test` are obtained by randomly splitting the 5,000 official SALICON validation images with a fixed seed (42).

| Split | Images | Source | Role |
| :--- | ---: | :--- | :--- |
| `train` | 10,000 | SALICON training set | Model training and B0 estimation |
| `tuning` | 2,500 | SALICON validation set | Development, model selection, and checkpoint selection |
| `internal_test` | 2,500 | SALICON validation set | Final post-freeze evaluation |

The official SALICON test set is not used for these quantitative comparisons because its ground-truth saliency maps are not publicly available. A 6,000-image training subset (`--dev_subset`) exists only for quick development runs; all official runs use the full training split.

### Preprocessing and Targets

All models share the same input pipeline, configured in `configs/data.yaml` and implemented in `src/data/dataset.py`:

- **Images** are converted to RGB, resized to 256 × 192 (width × height) with bilinear interpolation, and normalized with the ImageNet mean and standard deviation.
- **Saliency maps** are loaded as grayscale and resized to the same resolution with bilinear interpolation. Two versions of each map are produced:

| Target | Definition | Used by |
| :--- | :--- | :--- |
| `density_map_raw` | Grayscale values scaled to [0, 1] | Training of B1 and M1 (MSE loss) |
| `density_map_prob` | Raw map normalized to sum to 1: `(P + ε) / Σ(P + ε)`, with `ε = 1e-6` | Training of B0, M1-L, G and M2; evaluation metrics of all models |

- **Augmentation**: the only augmentation is a horizontal flip with probability 0.5, applied identically to the image and its maps. It is used on the training split only; `tuning` and `internal_test` are never augmented.
- **Fixations** are kept for the fixation-based metrics (NSS, sAUC) and are not used for training.

## Training and Evaluation Protocol

### Metrics

The primary metrics, reported for all models, are:

- **CC**: Pearson correlation coefficient between predicted and ground-truth saliency maps
- **SIM**: similarity (histogram intersection)
- **KLD**: Kullback-Leibler divergence, computed as `KLD(target || prediction)`

**Evaluation in probability space.** CC, SIM and KLD are always computed on predictions converted to probability maps (summing to 1), for all models: B0 already outputs a probability map, while B1, M1, M1-L, G and M2 go through `predict_probability()`. The training loss is unaffected (B1 and M1 still use MSE on the raw output). This keeps the metrics comparable across models, since CC is not exactly scale-invariant when the stabilizing epsilon (`1e-6`) is used.

**NSS** and **sAUC** are also implemented in `src/metrics.py`, but they require fixation coordinates and are not part of the main comparison. Per-image results from different models are always aligned by `image_id`; missing or duplicated IDs are treated as errors. The full protocol is described in `scripts/docs/evaluation_protocol.md`.

### Training Protocol

All trainable models (B1, M1, M1-L, G, M2) share the same frozen training settings for the official runs:

- full training split (10,000 images), without `--dev_subset`
- seed 42 (shared with the fixed split) and batch size 8
- AdamW optimizer, learning rate `1e-4`, weight decay `1e-4`
- up to 30 epochs, with early stopping (patience of 5 epochs)
- best checkpoint selected on `tuning` using CC (higher is better)
- a clean checkpoint directory for each model, because `train.py` automatically resumes from `<experiment>_last.pt` if it finds one

The 30 epochs are a maximum budget rather than a fixed length: early stopping can end a run sooner, and the model used for evaluation is the best checkpoint on `tuning`, not the last epoch. Training and validation loss curves are logged for diagnostic purposes only. The models optimize different objectives (MSE for B1 and M1, CC + KLD for M1-L, G and M2), so their loss values are not comparable; models are compared exclusively through CC, SIM and KLD on the same split.

**G** follows the same protocol, but only its gate is trained (32,897 parameters: a two-layer MLP, 512 → 64 → 1), while the M1-L base model and the B0 prior stay frozen.

Epoch of the selected checkpoint for each model:

| Model | Best epoch |
| :--- | ---: |
| B1 | 10 |
| M1 | 7 |
| M1-L | 7 |
| G | 1 |
| M2 | 9 |

Since the patience is 5 epochs, every run was ended by early stopping well before the 30-epoch limit.

Recommended run order: `B0 → B1 → M1 → M1-L → G → M2`. G requires the final M1-L checkpoint and the B0 center map, while M2 is independent and can be run at any point. Model and loss settings are defined in `configs/experiments.yaml`.

## Usage

All commands are run from the repository root. Values in `<...>` are paths to be replaced.

### Training

```bash
python scripts/train.py --experiment B0   --data_dir data/ --checkpoint_dir <ckpt_dir>
python scripts/train.py --experiment B1   --data_dir data/ --checkpoint_dir <ckpt_dir>
python scripts/train.py --experiment M1   --data_dir data/ --checkpoint_dir <ckpt_dir>
python scripts/train.py --experiment M1-L --data_dir data/ --checkpoint_dir <ckpt_dir>
python scripts/train.py --experiment M2   --data_dir data/ --checkpoint_dir <ckpt_dir>

python scripts/train.py --experiment G --data_dir data/ --checkpoint_dir <ckpt_dir> \
    --base_checkpoint <M1-L_best.pt> --center_prior_checkpoint <B0_center_map.pt>
```

B0 involves no backpropagation: the command computes the mean saliency map from the training set.

### Evaluation

```bash
python scripts/evaluate.py --experiment M1 --checkpoint_path <M1_best.pt> \
    --split tuning --data_dir data/ --results_dir <evaluation_dir>
```

This saves per-image results and a summary for the selected split. The checkpoint must declare the same `experiment` passed on the command line, otherwise it is rejected. Evaluating on `internal_test` additionally requires `--final_evaluation`, and should be done only once per model, after code, splits, seeds, and hyperparameters are frozen.

### Paired Comparisons and Confidence Intervals

```bash
python scripts/compare_evaluations.py \
    <evaluation_dir>/B1/tuning_per_image.csv \
    <evaluation_dir>/M1/tuning_per_image.csv \
    --left-name B1 --right-name M1 \
    --output <evaluation_dir>/comparisons/M1_minus_B1.csv

python scripts/bootstrap_ci.py <evaluation_dir>/comparisons/M1_minus_B1.csv \
    --n_resamples 1000 --seed 42 \
    --output <evaluation_dir>/comparisons/bootstrap_M1_minus_B1.json
```

The first command aligns two models image by image and computes per-image differences. The second estimates paired bootstrap confidence intervals on those differences. Repeat both for each comparison listed in [Model Variants and Comparisons](#model-variants-and-comparisons).

### Gate Diagnostics and Qualitative Figures

```bash
python scripts/diagnose_gate.py --checkpoint <G_best.pt> --data_dir data/

python scripts/qualitative_figures.py --data_dir data/ \
    --checkpoint_dir <ckpt_dir> --results_dir <evaluation_dir> \
    --output <figure.png>
```

`diagnose_gate.py` checks whether the gate of G actually varies across images or collapses to a near-constant value. `qualitative_figures.py` builds a side-by-side comparison of all six models on representative examples and needs all their checkpoints.

## Results

Quantitative results, statistical comparisons, and their discussion are reported in the project report.

## References

The project builds on previous work in visual saliency prediction, multi-scale deep architectures, saliency evaluation, and hierarchical vision models.

### Paper References

1. M. Jiang, S. Huang, J. Duan, and Q. Zhao, *SALICON: Saliency in Context*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015.

2. X. Huang, C. Shen, X. Boix, and Q. Zhao, *SALICON: Reducing the Semantic Gap in Saliency Prediction by Adapting Deep Neural Networks*. IEEE International Conference on Computer Vision (ICCV), 2015.

3. K. He, X. Zhang, S. Ren, and J. Sun, *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

4. S. S. S. Kruthiventi, V. Gudisa, J. H. Dholakiya, and R. V. Babu, *Saliency Unified: A Deep Architecture for Simultaneous Eye Fixation Prediction and Salient Object Segmentation*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

5. W. Wang and J. Shen, *Deep Visual Attention Prediction*. IEEE Transactions on Image Processing, vol. 27, no. 5, pp. 2368–2378, 2018.

6. J. Pan, C. Canton-Ferrer, K. McGuinness, N. E. O'Connor, J. Torres, E. Sayrol, and X. Giró-i-Nieto, *SalGAN: Visual Saliency Prediction with Generative Adversarial Networks*. arXiv:1701.01081, 2017 (shorter version presented at the CVPR 2017 Scene Understanding Workshop, SUNw).

7. Z. Bylinskii, T. Judd, A. Oliva, A. Torralba, and F. Durand, *What Do Different Evaluation Metrics Tell Us About Saliency Models?*. IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 3, pp. 740–757, 2019.

8. T. Judd, K. Ehinger, F. Durand, and A. Torralba, *Learning to Predict Where Humans Look*. IEEE International Conference on Computer Vision (ICCV), 2009.

9. N. Liu, N. Zhang, K. Wan, L. Shao, and J. Han, *Visual Saliency Transformer* (salient object detection). IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

10. J. Lou, H. Lin, D. Marshall, D. Saupe, and H. Liu, *TranSalNet: Towards Perceptually Relevant Visual Saliency Prediction*. Neurocomputing, vol. 494, pp. 455–467, 2022.

11. W. Wang, E. Xie, X. Li, D.-P. Fan, K. Song, D. Liang, T. Lu, P. Luo, and L. Shao, *PVT v2: Improved Baselines with Pyramid Vision Transformer*. Computational Visual Media, 2022.

12. W. Wang, Q. Lai, H. Fu, J. Shen, H. Ling, and R. Yang, *Salient Object Detection in the Deep Learning Era: An In-depth Survey* (salient object detection). IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022. DOI: 10.1109/TPAMI.2021.3051099.

13. A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS), 2019.

### Software and Data Resources

- **PyTorch Image Models (`timm`)** — R. Wightman. Used to provide the pretrained PVTv2-B1 backbone employed by M2.
- **SALICON dataset** — the project uses SALICON images, saliency maps, and fixation data; the dataset files used for the experiments were obtained through the Kaggle distribution.