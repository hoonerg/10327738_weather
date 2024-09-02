import gc
from fire import Fire
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl

from vqvae.vqvae import VQVAE
from diffusion import LatentDiffusion
from data_module import WeatherDataModule


def setup_model(
    num_timesteps=1000,
    vqvae_model_path="../vqvae/vqvae_model.pth",
    lr=1e-4,
    num_embeddings=1024,
    embedding_dim=16,
    model_dir="../test/"
):
    # Load the pre-trained VQ-VAE model
    vqvae_model = VQVAE(input_dim=89, num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    vqvae_model.load_state_dict(torch.load(vqvae_model_path))
    vqvae_model.requires_grad_(False)
    vqvae_model.eval()

    # Latent Diffusion Model with VQ-VAE
    model = LatentDiffusion(
        model=vqvae_model,
        vqvae_model_path=vqvae_model_path,
        timesteps=num_timesteps,
        lr=lr,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim
    )

    # Configure ModelCheckpoint callback for saving the model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_loss_ema:.4f}",
        monitor="val_loss_ema",
        save_top_k=3,
        mode='min',
        save_weights_only=True
    )

    # Configure PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count(),
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss_ema", patience=6),
            checkpoint_callback
        ]
    )

    gc.collect()
    return model, trainer


def train(
    num_timesteps=1000,
    batch_size=64,
    model_dir="../test/",
    vqvae_model_path="../models/vqvae/vqvae_model.pth",
    lr=1e-4
):
    print("Loading data...")
    datamodule = WeatherDataModule(
        data_dir="/scratch/n36837sc/projects/10327738_weather/dataset/processed",
        batch_size=batch_size
    )

    print("Setting up model...")
    model, trainer = setup_model(
        num_timesteps=num_timesteps,
        vqvae_model_path=vqvae_model_path,
        lr=lr,
        model_dir=model_dir
    )

    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)


def main(config=None, **kwargs):
    config = OmegaConf.load(config) if config is not None else {}
    config.update(kwargs)
    train(**config)


if __name__ == "__main__":
    Fire(main)