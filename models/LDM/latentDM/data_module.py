import os
import torch
from torch.utils.data import DataLoader, Dataset
import xarray as xr
import pytorch_lightning as pl

class WeatherDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the data file
        file_path = self.files[idx]
        with xr.open_dataset(file_path) as ds:
            data = ds.to_array().transpose("latitude", "longitude", "variable").values
            data = torch.tensor(data, dtype=torch.float32)  # Convert to tensor

        return data

class WeatherDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = WeatherDataset(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
