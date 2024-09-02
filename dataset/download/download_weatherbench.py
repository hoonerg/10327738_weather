import argparse
import xarray as xr
import os

def process_timestep(task_id):
    # Google Clound Bucket ----(zarr format)---> this server

    # Load the dataset
    ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr', consolidated=True)
    # higher resolution: 1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr
    ds_sel = ds.isel(time=task_id)
    
    # Output Derectory
    output_directory = f"../weatherbench/{ds_sel.time.dt.strftime('%Y').values}"
    filename = f"{output_directory}/data_{ds_sel.time.dt.strftime('%Y%m%d_%H').values}.nc"
    
    # Check directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save data
    ds_sel.to_netcdf(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, help="Task ID for this job array task")
    args = parser.parse_args()
    
    process_timestep(args.task_id)