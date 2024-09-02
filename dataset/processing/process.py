import xarray as xr
import numpy as np
import pandas as pd
import os
import pickle

# Load global statistics
global_stats_path = '/scratch/n36837sc/projects/10327738_weather/processing/global_stats.pkl'
with open(global_stats_path, 'rb') as f:
    global_stats = pickle.load(f)

# Define directories
raw_base_directory = '/scratch/n36837sc/projects/10327738_weather/weatherbench/raw/'
processed_base_directory = '/scratch/n36837sc/projects/10327738_weather/weatherbench/processed/'
static_variables_path = '/scratch/n36837sc/projects/10327738_weather/processing/surface.npy'  # Path to surface.npy

# Define variables
atmospheric_variables = [
    "geopotential", "temperature", "u_component_of_wind", "v_component_of_wind",
    "vertical_velocity", "specific_humidity"
]
surface_variables = [
    "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "sea_surface_temperature",
    "mean_sea_level_pressure", "total_cloud_cover", "total_precipitation_6hr"
]

# Define time range
start_date = '1971-01-01'
end_date = '2023-01-01'
time_freq = '6H'
time_range = pd.date_range(start=start_date, end=end_date, freq=time_freq)

# Load static variables from surface.npy
static_data = np.load(static_variables_path)  # Assumes shape (4, height, width)
static_variable_names = ['static_var1', 'static_var2', 'static_var3', 'static_var4']  # Replace with actual names

# Filter dataset for desired variables
def filter_dataset(ds, atmospheric_vars, surface_vars):
    variables_to_keep = atmospheric_vars + surface_vars
    return ds[variables_to_keep]

# Preprocess data function
def preprocess_data(ds, global_stats, atmospheric_vars, surface_vars):
    for variable in ds.data_vars:
        if variable in atmospheric_vars and 'level' in ds[variable].dims:  # Atmospheric variables
            for level in ds[variable].level:
                level_item = level.item()
                if variable in global_stats and level_item in global_stats[variable]:
                    mean, std = global_stats[variable][level_item]
                    ds[variable].loc[dict(level=level)] = (ds[variable].sel(level=level) - mean) / std
        elif variable in surface_vars:  # Surface variables
            if variable in global_stats:
                mean, std = global_stats[variable]
                ds[variable] = (ds[variable] - mean) / std
    return ds

# Add static variables to the dataset
def add_static_variables(ds, static_data, static_variable_names):
    height, width = ds.dims['latitude'], ds.dims['longitude']  # Assuming these are the correct dimensions
    for i, var_name in enumerate(static_variable_names):
        ds[var_name] = (('latitude', 'longitude'), static_data[i, :height, :width])
    return ds

# Process each file
for timestamp in time_range:
    file_name = f"data_{timestamp.strftime('%Y%m%d_%H')}.nc"
    raw_file_path = os.path.join(raw_base_directory, timestamp.strftime('%Y'), file_name)
    processed_file_path = os.path.join(processed_base_directory, timestamp.strftime('%Y'), file_name)

    if os.path.exists(raw_file_path):
        with xr.open_dataset(raw_file_path) as ds:
            # Filter and preprocess
            ds_filtered = filter_dataset(ds, atmospheric_variables, surface_variables)
            ds_preprocessed = preprocess_data(ds_filtered, global_stats, atmospheric_variables, surface_variables)
            
            # Add static variables
            ds_final = add_static_variables(ds_preprocessed, static_data, static_variable_names)
            
            # Save preprocessed dataset
            os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
            ds_final.to_netcdf(processed_file_path)
