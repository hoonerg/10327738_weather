import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
from ....models.LDM.vqvae.vqvae import VQVAE
from ....models.LDM.latentDM.diffusion import LatentDiffusion
from sklearn.metrics import mean_squared_error

# Directories
processed_base_directory = '/scratch/n36837sc/projects/10327738_weather/weatherbench/processed/'
output_base_directory = '/scratch/n36837sc/projects/10327738_weather/weatherbench/sampling/samples/'
label_base_directory = '/scratch/n36837sc/projects/10327738_weather/weatherbench/sampling/label/'

# Time range
start_date = '1971-01-01'
end_date = '2023-01-01'
time_freq = '6H'
time_range = pd.date_range(start=start_date, end=end_date, freq=time_freq)

# Load models
def load_model(vqvae_model_path, diffusion_model_path, num_embeddings=1024, embedding_dim=16, num_timesteps=1000):
    # Load VQ-VAE
    vqvae_model = VQVAE(input_dim=89, num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    vqvae_model.load_state_dict(torch.load(vqvae_model_path))
    vqvae_model.requires_grad_(False)
    vqvae_model.eval()

    # Load Latent Diffusion
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

# Compute ranks and save labels
def compute_rank_and_save_labels(samples, ground_truth, timestamp, label_dir):
    # Calculate MSE scores
    scores = [mean_squared_error(ground_truth.flatten(), sample.flatten()) for sample in samples]
    # Rank samples
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

# Generate and save samples
def generate_and_save_samples(model, ds, timestamp, output_dir, label_dir, device='cpu'):
    model.to(device)
    # Prepare input
    input_data = ds.to_array().transpose("latitude", "longitude", "variable").values
    input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        latent_input = model.encode(input_data)
        samples = []
        for i in range(8):
            # Generate sample
            noise = torch.randn_like(latent_input).to(device)
            z_combined = torch.cat((latent_input, noise), dim=1)
            predicted_latent = model.apply_model(z_combined, t=torch.tensor([model.num_timesteps - 1]).to(device))
            prediction = model.decode(predicted_latent).cpu().numpy()
            samples.append(prediction)

            # Save sample
            output_path = os.path.join(output_dir, f"sample_{timestamp.strftime('%Y%m%d_%H')}_sample_{i+1}.npy")
            np.save(output_path, prediction)
            print(f"Saved sample: {output_path}")

        # Rank samples and save labels
        compute_rank_and_save_labels(samples, input_data.cpu().numpy(), timestamp, label_dir)

# Main function
def main(
    vqvae_model_path="../../models/LDM/vqvae/vqvae_model.pth",
    diffusion_model_path="../../models/LDM/test/best_model.ckpt",
    num_embeddings=1024,
    embedding_dim=16,
    num_timesteps=1000
):
    # Load models
    model = load_model(vqvae_model_path, diffusion_model_path, num_embeddings, embedding_dim, num_timesteps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for timestamp in time_range:
        # File paths
        file_name = f"data_{timestamp.strftime('%Y%m%d_%H')}.nc"
        processed_file_path = os.path.join(processed_base_directory, timestamp.strftime('%Y'), file_name)
        output_dir = os.path.join(output_base_directory, timestamp.strftime('%Y'))
        label_dir = os.path.join(label_base_directory, timestamp.strftime('%Y'))

        if os.path.exists(processed_file_path):
            os.makedirs(output_dir, exist_ok=True)
            with xr.open_dataset(processed_file_path) as ds:
                # Generate and save samples
                generate_and_save_samples(model, ds, timestamp, output_dir, label_dir, device)

if __name__ == "__main__":
    main()
