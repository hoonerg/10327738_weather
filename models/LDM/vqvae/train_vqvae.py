# vqvae/train_vqvae.py

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from vqvae.vqvae import VQVAE
from vqvae.data_loader import WeatherDataset
import argparse
import yaml

# Argument parser for config file
parser = argparse.ArgumentParser(description='Train VQ-VAE model.')
parser.add_argument('--config', type=str, default='params.yml', help='Path to the YAML configuration file')
args = parser.parse_args()

# Load parameters from the YAML configuration file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Extract parameters
epochs = config['epochs']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
data_dir = config['data_dir']
num_embeddings = config['num_embeddings']
embedding_dim = config['embedding_dim']

# Load Data
dataset = WeatherDataset(base_directory=data_dir)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer with parameters from the config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VQVAE(input_dim=89, num_embeddings=num_embeddings, embedding_dim=embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(epochs):
    total_loss = 0
    for data in data_loader:
        data = data.to(device)  # Shape: (batch_size, 64, 32, 89)
        optimizer.zero_grad()
        
        recon_data, vq_loss = model(data)
        recon_loss = F.mse_loss(recon_data, data)
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(data_loader)}")

torch.save(model.state_dict(), 'vqvae_model.pth')