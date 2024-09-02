
# 10327738 MSc Dissertation

This project aims to develop and train a weather prediction model using a combination of VQ-VAE, Latent Diffusion Models, and a custom processor. The model leverages various weather datasets to compute derived variables and predict future weather conditions.

[View the full PDF figure](figures/model_architecture.pdf)
[View the full PDF figure](figures/physics.pdf)

## Directory Structure

The project is organized as follows:

```
10327738_weather/
│
├── dataset/                 # Contains weather-related data
│   ├── download/            # Scripts for downloading datasets
│   ├── processing/          # Scripts for preprocessing datasets
│   ├── sampling/            # Contains samples and rank labels
│   └── weatherbench/        # Datasets
│
├── models/                  # Contains model-related code
│   └── LDM/                 # Latent Diffusion Model
│       ├── blocks/          # Building blocks for the LDM
│       ├── latentDM/        # Latent Diffusion Model scripts
│       ├── test/            # Testing scripts for the models
│       └── vqvae/           # VQ-VAE model scripts
│       └── utils.py         # Utility functions for models
│
├── sampling/                # Sampling-related scripts
│   ├── labelling/           # Scripts for generating labels
│   └── physics/             # Physics-based derived variable computation
│
└── processor/               # Processor model scripts for post-processing and ranking
```

## Setup

### Step 1: Create the Conda Environment

Ensure you have `conda` installed. Then, create the environment with all necessary dependencies by running:

```bash
conda env create -f environment.yml
conda activate ldm
```

### Step 2: Download and Preprocess Data

1. **Download data**: Use the scripts in `dataset/download/` to fetch the required datasets. 
Note that the total size of dataset is 2.4Tb.
   
2. **Preprocess data**: Run the preprocessing scripts in `dataset/processing/` to prepare the data for training.

### Step 3: Train the Model

1. **Train the VQ-VAE model**: Use the scripts in `models/LDM/vqvae/` to train the VQ-VAE model on the preprocessed data.

2. **Train the Latent Diffusion Model**: After training the VQ-VAE, use the scripts in `models/LDM/latentDM/` to train the Latent Diffusion Model using the VQ-VAE encoded data.

3. **Train the Processor Model**: Finally, train the processor model to rank the samples using the script in `processor/`.

```bash
python train_processor.py
```

### Step 4: Inference and Evaluation

Use the inference scripts to generate predictions and rank the samples:

```bash
python inference.py
```

## Physics-Based Derived Variables

Derived variables like divergence, vorticity, total column water vapor, and integrated vapor transport are computed using scripts in `sampling/physics/`. These derived variables are critical for training the processor to correctly rank the samples based on their similarity to the ground truth. 

## Important Files

- **`environment.yml`**: Defines the Python environment and dependencies for this project.
- **`train_processor.py`**: Main script for training the processor model.
- **`inference.py`**: Script for generating predictions and evaluating the model.
- **`physics.py`**: Contains the `PhysicsModel` class used for computing derived variables.
- **`vqvae.py`**: Contains the VQ-VAE model implementation and training logic.
- **`latentDM.py`**: Contains the Latent Diffusion Model implementation.

## Acknowledgments
This project is based on code from the following repositories:

https://github.com/MeteoSwiss/ldcast
https://github.com/NVlabs/edm
https://github.com/gaozhihan/PreDiff/tree/main
