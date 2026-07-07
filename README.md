# U-Net Based Stress Field Prediction for Cantilever Beam Structures

This project applies deep learning to predict von Mises stress fields in cantilever beam lattice structures from strut-thickness / volume fraction distributions. It also explores the inverse problem of reconstructing volume fraction distributions from stress fields.

## Project Overview

The underlying structure is a cantilever beam made of centered rectangular unit cells. Each strut has a different thickness value, and the objective is to understand how thickness distribution affects the resulting von Mises stress field.

This project implements two models:

- **Forward U-Net:** Predicts the normalized von Mises stress field from a volume fraction / strut-thickness distribution.
- **Inverse U-Net:** Predicts the volume fraction distribution from a normalized stress field.

## Dataset Description

The dataset contains:

- `output.xlsx`: Volume fraction / strut-thickness data for different material distributions.
- `stress/`: Text files containing von Mises stress values and deformed coordinates.
- `cord.txt`: Coordinates of the undeformed structure.

In the implemented pipeline:

- Each input volume distribution has **226 thickness values**.
- Volume vectors are padded and converted into **64 x 64 image-like arrays**.
- Each stress field contains **3304 stress values**.
- Stress vectors are reshaped and padded into **64 x 64 stress-field arrays**.

## Methodology

1. Loaded volume fraction data from `output.xlsx`.
2. Loaded von Mises stress values from the `stress/` folder.
3. Converted volume and stress vectors into image-like 2D representations.
4. Applied min-max normalization to volume distributions.
5. Applied log transformation and normalization to stress fields.
6. Split the dataset into training, validation, and testing sets.
7. Trained a Forward U-Net for stress-field prediction.
8. Trained an Inverse U-Net for volume-distribution reconstruction.
9. Evaluated model performance using MSE, MAE, training/validation loss curves, and absolute error plots.

## Model Architecture

The U-Net architecture consists of:

- Encoder blocks using `Conv2D` and `MaxPooling2D`
- Bottleneck convolution layers
- Decoder blocks using `Conv2DTranspose`
- Skip connections between encoder and decoder layers
- Final `Conv2D` layer with sigmoid activation

Both the Forward and Inverse U-Nets use the same architecture.

## Tools and Libraries

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Results

The Forward U-Net was trained to learn the mapping from material distribution to normalized von Mises stress field. The Inverse U-Net was trained to reconstruct the volume fraction distribution from the stress field.

The project compares the two tasks using:

- Training loss
- Validation loss
- MSE
- MAE
- True vs predicted field plots
- Absolute error plots

The Forward U-Net showed more stable convergence, while the Inverse U-Net was more complex because multiple material distributions can produce similar stress-field patterns.

## Repository Structure

```text
U-Net-Based-Stress-Field-Prediction-for-Cantilever-Beam-Structures/
│
├── src/
│   └── unet_stress_prediction.py
│
├── results/
│   ├── forward_unet_loss.png
│   ├── inverse_unet_loss.png
│   ├── forward_stress_true.png
│   ├── forward_stress_predicted.png
│   ├── forward_stress_absolute_error.png
│   ├── inverse_volume_true.png
│   ├── inverse_volume_predicted.png
│   └── inverse_volume_absolute_error.png
│
├── report/
│   └── project_report.pdf
│
└── README.md
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/prakhargarg0106/U-Net-Based-Stress-Field-Prediction-for-Cantilever-Beam-Structures.git
```

2. Move into the project folder:

```bash
cd U-Net-Based-Stress-Field-Prediction-for-Cantilever-Beam-Structures
```

3. Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow openpyxl
```

4. Place the dataset folder or zip file in the project directory:

```text
prob1_data/
```

or

```text
prob1_data(1).zip
```

5. Run the script:

```bash
python src/unet_stress_prediction.py
```

## Note on Dataset

The dataset is not included in this repository because it is too large. The repository contains the implementation code, methodology, and result visualizations.

## Project Status

Completed as part of a Deep Learning for Physical Systems project.

## Author

**Prakhar Garg**  
B.Tech Mechanical Engineering  
Indian Institute of Technology Ropar
