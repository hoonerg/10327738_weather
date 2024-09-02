import os
import torch
import numpy as np
from processor import ProcessorModel
from ..physics.physics_processor import PhysicsModel  # Import PhysicsModel class
from utils import load_model, compute_rank_and_save_labels

def train_processor(
    condition_data_path,
    sample_data_path,
    label_data_path,
    save_model_path="processor_model.pth",
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = ProcessorModel().to(device)
    optimizer = torch.optim.Adam(processor.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Instantiate the PhysicsModel
    physics_model = PhysicsModel()

    # Prepare data (condition, samples, and labels)
    condition_files = sorted([os.path.join(condition_data_path, f) for f in os.listdir(condition_data_path) if f.endswith('.npy')])
    sample_files = sorted([os.path.join(sample_data_path, f) for f in os.listdir(sample_data_path) if f.endswith('.npy')])
    label_files = sorted([os.path.join(label_data_path, f) for f in os.listdir(label_data_path) if f.endswith('.npy')])

    for epoch in range(num_epochs):
        total_loss = 0

        for i in range(0, len(condition_files), batch_size):
            # Load data in batches
            condition_batch = [np.load(f) for f in condition_files[i:i + batch_size]]
            sample_batch = [np.load(f) for f in sample_files[i * 8:(i + batch_size) * 8]]  # 8 samples per condition
            label_batch = [np.load(f) for f in label_files[i:i + batch_size]]

            condition_batch = torch.tensor(np.stack(condition_batch)).to(device)
            sample_batch = torch.tensor(np.stack(sample_batch)).view(-1, 64, 32, 117).to(device)
            label_batch = torch.tensor(np.concatenate(label_batch)).to(device)

            # Compute physics output
            physics_output = physics_model.compute(condition_batch.cpu().numpy())
            physics_output = torch.tensor(physics_output).to(device)
            condition_concat = torch.cat((condition_batch, physics_output), dim=-1)

            sample_concat = []
            for j in range(8):
                sample_data = sample_batch[j::8]
                physics_output_sample = physics_model.compute(sample_data.cpu().numpy())
                physics_output_sample = torch.tensor(physics_output_sample).to(device)
                sample_concat.append(torch.cat((sample_data, physics_output_sample), dim=-1))
            sample_concat = torch.stack(sample_concat).view(-1, 64, 32, 117)

            # Forward pass through the processor model
            condition_embedding = processor(condition_concat)
            sample_embeddings = processor(sample_concat)

            # Compute Euclidean distances between condition and each sample
            distances = torch.cdist(sample_embeddings, condition_embedding.unsqueeze(0), p=2).squeeze()

            # Compute the predicted rank by sorting distances
            predicted_ranks = torch.argsort(distances, dim=1)

            # Compute loss
            loss = criterion(predicted_ranks, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(condition_files)}")

        # Save the model
        torch.save(processor.state_dict(), save_model_path)
        print(f"Model saved at {save_model_path}")

if __name__ == "__main__":
    train_processor(
        condition_data_path='/scratch/n36837sc/projects/10327738_weather/weatherbench/processed/',
        sample_data_path='/scratch/n36837sc/projects/10327738_weather/weatherbench/samples/',
        label_data_path='/scratch/n36837sc/projects/10327738_weather/weatherbench/sampling/label/'
    )
