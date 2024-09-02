import xarray as xr
import numpy as np
import os

# Path to a dataset file
sample_file_path = '/scratch/n36837sc/projects/10327738_weather/weatherbench/raw/1971/data_19710101_00.nc'  # Update with actual path

# Output path for surface.npy
output_path = '/scratch/n36837sc/projects/10327738_weather/processing/surface.npy'

# Load the dataset
with xr.open_dataset(sample_file_path) as ds:
    # Extract variables and coordinates
    geopotential_at_surface = ds['geopotential_at_surface']
    land_sea_mask = ds['land_sea_mask']
    latitude = ds.coords['latitude']
    longitude = ds.coords['longitude']
    
    # Normalise by max value
    geopotential_norm = geopotential_at_surface / geopotential_at_surface.max()
    land_sea_mask_norm = land_sea_mask / land_sea_mask.max()
    
    # Create 2D fields for coordinates
    latitude_2d = np.tile(latitude.values[:, np.newaxis], (1, longitude.size))
    longitude_2d = np.tile(longitude.values[np.newaxis, :], (latitude.size, 1))
    
    # Normalise coordinate fields
    latitude_norm = latitude_2d / latitude_2d.max()
    longitude_norm = longitude_2d / longitude_2d.max()
    
    # Combine into 4-channel array
    surface_data = np.stack([geopotential_norm.values, land_sea_mask_norm.values, latitude_norm, longitude_norm], axis=-1)

    # Save as .npy file
    np.save(output_path, surface_data)
