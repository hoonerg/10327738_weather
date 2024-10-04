
# Latent Diffusion Model with VQVAE for Global Weather Forecasting

![Model_structure-1](https://github.com/user-attachments/assets/1041d553-3133-49b6-a519-479815e4a900)
![physics-1](https://github.com/user-attachments/assets/d9134f86-0779-42ff-96ec-8ae02d891923)

## Abstract
Accurate weather forecasting is crucial for various industries and public safety. Tradition- ally, it relies on Numerical Weather Prediction (NWP) models, which demand significant computational resources and are constrained by the inherent unpredictability of atmo- spheric dynamics. Recent research has increasingly explored the use of deep learning techniques for weather forecasting, with some models even outperforming traditional op- erational NWP models across various metrics. Accordingly, this study explores the use of deep learning models, specifically a physics-informed Latent Diffusion Model (LDM), to im- prove global weather forecasting. The proposed LDM combines a Vector-Quantised Varia- tional Autoencoder (VQVAE) to encode high-dimensional atmospheric data into a compact latent space with a diffusion model that predicts future atmospheric states. A novel sample selection model further enhances forecast accuracy by identifying the most physically plau- sible samples using additional physical variables. Experimental results demonstrate that the LDM outperforms baseline models, such as UNet and Persistence, for key atmospheric variables. In addition, the inclusion of physical information and the sample selection model significantly contribute to reducing errors and improving the reliability of the predictions. However, it does not yet match the performance of state-of-the-art models, indicating the need for further refinements. Future research should focus on refining model architecture, incorporating more explicit physical information, and expanding the dataset to improve ac- curacy and generalisation.

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

## Acknowledgments
This project is based on code from the following repositories:

https://github.com/MeteoSwiss/ldcast  
https://github.com/NVlabs/edm  
https://github.com/gaozhihan/PreDiff/tree/main  
The author would like to acknowledge the assistance given by Research IT and the use of the Computational Shared Facility at The University of Manchester.
