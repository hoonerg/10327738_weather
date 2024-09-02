import torch
from torch.utils.data import Dataset
import xarray as xr
import os
import glob

class WeatherDataset(Dataset):
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.data_files = self._get_all_files()

    def _get_all_files(self):
        # Find all .nc files in the processed directory for each year
        all_files = []
        for year in range(1959, 2024):  # Adjust the range as needed
            year_dir = os.path.join(self.base_directory, str(year))
            files = glob.glob(os.path.join(year_dir, "data_*.nc"))
            all_files.extend(files)
        return all_files

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load the dataset file
        file_path = self.data_files[idx]
        with xr.open_dataset(file_path) as ds:
            # Convert dataset variables to a numpy array and reshape
            data = ds.to_array().transpose("latitude", "longitude", "variable").values
            data = torch.tensor(data, dtype=torch.float32)  # Shape: (64, 32, 89)

        return data