import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial

from .utils import make_beta_schedule, extract_into_tensor, noise_like, timestep_embedding
from .ema import LitEma
from vqvae.vqvae import VQVAE

class LatentDiffusion(pl.LightningModule):
    # parameters
    def __init__(self,
        model,
        vqvae_model_path,
        context_encoder=None,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        use_ema=True,
        lr=1e-4,
        lr_warmup=0,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",
        num_embeddings=1024,
        embedding_dim=16,
    ):
        super().__init__()
        self.model = model
        
        # Load and configure the VQ-VAE autoencoder
        self.autoencoder = VQVAE(input_dim=89, num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.autoencoder.load_state_dict(torch.load(vqvae_model_path))
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()

        self.conditional = (context_encoder is not None)
        self.context_encoder = context_encoder
        self.lr = lr
        self.lr_warmup = lr_warmup

        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.register_schedule(
            beta_schedule=beta_schedule, timesteps=timesteps,
            linear_start=linear_start, linear_end=linear_end, 
            cosine_s=cosine_s
        )

        self.loss_type = loss_type

    def encode(self, x):
        # Encode each spatial coordinate separately
        batch_size, height, width, _ = x.shape  # (batch_size, 64, 32, 89)
        z_e = self.autoencoder.encoder(x.view(batch_size * height * width, -1))
        z_q, _, _ = self.autoencoder.vector_quantizer(z_e)
        z_q = z_q.view(batch_size, height, width, -1)  # Reshape back
        return z_q

    def decode(self, z):
        # Decode each spatial coordinate separately
        batch_size, height, width, _ = z.shape  # (batch_size, 64, 32, embedding_dim)
        x_recon = self.autoencoder.decoder(z.view(batch_size * height * width, -1))
        x_recon = x_recon.view(batch_size, height, width, -1)  # Reshape back
        return x_recon

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = make_beta_schedule(
            beta_schedule, timesteps,
            linear_start=linear_start, linear_end=linear_end,
            cosine_s=cosine_s
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def apply_model(self, x_noisy, t, cond=None, return_ids=False):
        if self.conditional:
            cond = self.context_encoder(cond)
        with self.ema_scope():
            return self.model(x_noisy, t, context=cond)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None, context=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        # Encode condition
        z_cond = self.encode(context)
        z_t = noise_like(x_start.shape, device=x_start.device)
        z_combined = torch.cat((z_cond, z_t), dim=1)  # Concatenate along channel dimension

        # Diffusion model
        model_out = self.model(z_combined, t, context=context)

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        return self.get_loss(model_out, target, mean=False).mean()

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def shared_step(self, batch):
        (x, y) = batch
        y = self.encode(y)  # Use VQ-VAE encoder to get latent space
        context = self.context_encoder(x) if self.conditional else None
        return self(y, context=context)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        with self.ema_scope():
            loss_ema = self.shared_step(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("val_loss", loss, **log_params)
        self.log("val_loss_ema", loss, **log_params)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
            betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_loss_ema",
                "frequency": 1,
            },
        }

    def optimizer_step(
        self, 
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        **kwargs    
    ):
        if self.trainer.global_step < self.lr_warmup:
            lr_scale = (self.trainer.global_step+1) / self.lr_warmup
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        super().optimizer_step(
            epoch, batch_idx, optimizer,
            optimizer_idx, optimizer_closure,
            **kwargs
        )
