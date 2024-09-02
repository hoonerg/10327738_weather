import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=89, hidden_dims=[64, 32, 24, 16]):
        super(Encoder, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, height, width, _ = x.shape  # Shape: (batch_size, 64, 32, 89)
        x = x.view(batch_size * height * width, -1)  # Reshape to (batch_size * 64 * 32, 89)
        encoded = self.encoder(x)
        return encoded.view(batch_size, height, width, -1)  # Reshape back to (batch_size, 64, 32, hidden_dims[-1])


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=16, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        batch_size, height, width, channels = z.shape  # Shape: (batch_size, 64, 32, 16)
        flat_z = z.view(-1, self.embedding_dim)  # Flatten to (batch_size * 64 * 32, 16)
        
        # Compute distances
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_z, self.embedding.weight.t()))
        
        # Get closest embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = F.embedding(encoding_indices, self.embedding.weight).view(z.shape)
        
        # Loss for the embedding
        loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(z_q, z.detach())
        
        # Replace gradients
        z_q = z + (z_q - z).detach()
        return z_q, loss, encoding_indices.view(batch_size, height, width)


class Decoder(nn.Module):
    def __init__(self, output_dim=89, hidden_dims=[16, 24, 32, 64]):
        super(Decoder, self).__init__()
        layers = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        batch_size, height, width, _ = x.shape  # Shape: (batch_size, 64, 32, 16)
        x = x.view(batch_size * height * width, -1)  # Reshape to (batch_size * 64 * 32, 16)
        decoded = self.decoder(x)
        return decoded.view(batch_size, height, width, -1)  # Reshape back to (batch_size, 64, 32, 89)


class VQVAE(nn.Module):
    def __init__(self, input_dim=89, hidden_dims=[64, 32, 24, 16], num_embeddings=1024, embedding_dim=16):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(input_dim, list(reversed(hidden_dims)))

    def forward(self, x):
        z_e = self.encoder(x)  # Encode the input
        z_q, vq_loss, _ = self.vector_quantizer(z_e)  # Quantize the latent representation
        x_recon = self.decoder(z_q)  # Decode the quantized representation
        return x_recon, vq_loss