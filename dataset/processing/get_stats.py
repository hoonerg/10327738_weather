import xarray as xr
import numpy as np
import pandas as pd
import os
import pickle

base_directory = '/scratch/n36837sc/projects/10327738_weather/weatherbench/'

start_date = '1971-01-01'
end_date = '2023-01-01'
time_freq = '6H'  # Data every 6 hours (00, 06, 12, 18)

time_range = pd.date_range(start=start_date, end=end_date, freq=time_freq)

global_sums = {}
global_sums_sq = {}
global_counts = {}

def update_stats(data, sums, sums_sq, counts, var_name, level=None):
    if level is not None:
        if var_name not in sums:
            sums[var_name] = {}
            sums_sq[var_name] = {}
            counts[var_name] = {}
        if level not in sums[var_name]:
            sums[var_name][level] = data.sum()
            sums_sq[var_name][level] = (data**2).sum()
            counts[var_name][level] = data.count()
        else:
            sums[var_name][level] += data.sum()
            sums_sq[var_name][level] += (data**2).sum()
            counts[var_name][level] += data.count()
    else:
        if var_name not in sums:
            sums[var_name] = data.sum()
            sums_sq[var_name] = (data**2).sum()
            counts[var_name] = data.count()
        else:
            sums[var_name] += data.sum()
            sums_sq[var_name] += (data**2).sum()
            counts[var_name] += data.count()

# Load and process each file corresponding to each timestamp
for timestamp in time_range:
    #print(timestamp)
    file_name = f"data_{timestamp.strftime('%Y%m%d_%H')}.nc"
    file_path = os.path.join(base_directory, timestamp.strftime('%Y'), file_name)
    if os.path.exists(file_path):  # Check if the file exists
        with xr.open_dataset(file_path) as ds:
            for variable in ds.data_vars:
                if 'level' in ds[variable].dims:
                    for level in ds[variable].level:
                        data = ds[variable].sel(level=level)
                        update_stats(data, global_sums, global_sums_sq, global_counts, variable, level.item())
                else:
                    data = ds[variable]
                    update_stats(data, global_sums, global_sums_sq, global_counts, variable)

# Calculate final global mean and std deviation
global_stats = {}
for var_name, levels in global_sums.items():
    if isinstance(levels, dict):
        global_stats[var_name] = {}
        for level, sum_val in levels.items():
            mean_val = global_sums[var_name][level] / global_counts[var_name][level]
            var_val = (global_sums_sq[var_name][level] / global_counts[var_name][level]) - (mean_val**2)
            std_val = np.sqrt(var_val)
            global_stats[var_name][level] = (mean_val.values, std_val.values)
    else:
        mean_val = global_sums[var_name] / global_counts[var_name]
        var_val = (global_sums_sq[var_name] / global_counts[var_name]) - (mean_val**2)
        std_val = np.sqrt(var_val)
        global_stats[var_name] = (mean_val.values, std_val.values)

# Save the global_stats dictionary to a pickle file
output_pickle_path = '/scratch/n36837sc/projects/10327738_weather/processing/global_stats.pkl'
with open(output_pickle_path, 'wb') as f:
    pickle.dump(global_stats, f)

print(f"Statistics dictionary has been saved to {output_pickle_path}")