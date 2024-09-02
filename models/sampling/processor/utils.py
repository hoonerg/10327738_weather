# utils.py

import os
import torch
import numpy as np
from sklearn.metrics import mean_squared_error

def load_model(vqvae_model_path, diffusion_model_path, num_embeddings=1024, embedding_dim=16, num_timesteps=1000):
    from ....models.LDM.vqvae.vqvae import VQVAE
    from ....models.LDM.latentDM.diffusion import LatentDiffusion

    # Load VQ-VAE model
    vqvae_model = VQVAE(input_dim=89, num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    vqvae_model.load_state_dict(torch.load(vqvae_model_path))
    vqvae_model.requires_grad_(False)
    vqvae_model.eval()

    # Load diffusion model
    model = LatentDiffusion(
        model=vqvae_model,
        vqvae_model_path=vqvae_model_path,
        timesteps=num_timesteps,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim
    )
    model.load_state_dict(torch.load(diffusion_model_path))
    model.eval()

    return model

def compute_rank_and_save_labels(samples, ground_truth, timestamp, label_dir):
    # Compute MSE scores
    scores = [mean_squared_error(ground_truth.flatten(), sample.flatten()) for sample in samples]
    # Rank samples (lower MSE is better)
    ranks = np.argsort(scores)
    # Create labels for cross-entropy loss
    labels = np.zeros(len(samples), dtype=int)
    for i, rank in enumerate(ranks):
        labels[rank] = i
    # Save labels
    os.makedirs(label_dir, exist_ok=True)
    label_path = os.path.join(label_dir, f"label_{timestamp.strftime('%Y%m%d_%H')}.npy")
    np.save(label_path, labels)
    print(f"Saved labels: {label_path}")
