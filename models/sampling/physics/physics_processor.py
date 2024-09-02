# physics.py

import numpy as np

class PhysicsModel:
    def __init__(self):
        pass

    def compute_divergence(self, u, v):
        """Compute divergence from u and v wind components level by level."""
        divergence = []
        for level in range(u.shape[-1]):  # Loop over each level
            dudx = np.gradient(u[:, :, level], axis=1)  # Gradient along x-axis (longitude)
            dvdy = np.gradient(v[:, :, level], axis=0)  # Gradient along y-axis (latitude)
            divergence.append(dudx + dvdy)  # Compute divergence for each level
        return np.stack(divergence, axis=-1)  # Stack results to get (height, width, levels)

    def compute_vorticity(self, u, v):
        """Compute vorticity from u and v wind components level by level."""
        vorticity = []
        for level in range(u.shape[-1]):  # Loop over each level
            dudy = np.gradient(u[:, :, level], axis=0)  # Gradient along y-axis (latitude)
            dvdx = np.gradient(v[:, :, level], axis=1)  # Gradient along x-axis (longitude)
            vorticity.append(dvdx - dudy)  # Compute vorticity for each level
        return np.stack(vorticity, axis=-1)  # Stack results to get (height, width, levels)

    def compute_total_column_vapor(self, specific_humidity):
        """Compute total column water vapor by summing specific humidity vertically."""
        total_column_vapor = np.sum(specific_humidity, axis=-1)  # Sum over levels
        return total_column_vapor

    def compute_integrated_vapor_transport(self, u, v, specific_humidity):
        """Compute integrated vapor transport (IVT) level by level."""
        u_vapor = []
        v_vapor = []
        for level in range(specific_humidity.shape[-1]):  # Loop over each level
            u_vapor.append(u[:, :, level] * specific_humidity[:, :, level])
            v_vapor.append(v[:, :, level] * specific_humidity[:, :, level])
        
        u_vapor_sum = np.sum(u_vapor, axis=0)  # Sum vertically to get total transport in u
        v_vapor_sum = np.sum(v_vapor, axis=0)  # Sum vertically to get total transport in v
        ivt = np.sqrt(u_vapor_sum ** 2 + v_vapor_sum ** 2)
        return ivt

    def compute_derived_variables(self, input_data):
        """Compute the four derived variables and concatenate into a single output array."""
        # Extract required data from input channels
        temperature = input_data[..., 13:26]  # Channels 13-25 (13 levels)
        geopotential = input_data[..., 0:13]  # Channels 0-12 (13 levels)
        u_wind = input_data[..., 26:39]  # Channels 26-38 (13 levels)
        v_wind = input_data[..., 39:52]  # Channels 39-51 (13 levels)
        specific_humidity = input_data[..., 65:78]  # Channels 65-77 (13 levels)

        # Add dummy fields
        specific_cloud_liquid_water_content = np.ones((input_data.shape[0], input_data.shape[1], 1))
        specific_cloud_ice_water_content = np.ones((input_data.shape[0], input_data.shape[1], 1))
        wind_10m_u = np.ones((input_data.shape[0], input_data.shape[1], 1))
        wind_10m_v = np.ones((input_data.shape[0], input_data.shape[1], 1))
        temperature_2m = np.ones((input_data.shape[0], input_data.shape[1], 1))

        # Compute derived variables level by level
        divergence = self.compute_divergence(u_wind, v_wind)
        vorticity = self.compute_vorticity(u_wind, v_wind)
        total_column_vapor = self.compute_total_column_vapor(specific_humidity)
        integrated_vapor_transport = self.compute_integrated_vapor_transport(u_wind, v_wind, specific_humidity)

        # Concatenate all derived variables into a single output array (64x32x28)
        derived_output = np.concatenate(
            [divergence, vorticity, total_column_vapor[..., np.newaxis], integrated_vapor_transport[..., np.newaxis]], axis=-1
        )

        return derived_output

    def compute(self, input_data):
        """Compute physics output for input data."""
        return self.compute_derived_variables(input_data)
