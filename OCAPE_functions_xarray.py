import netCDF4 as nc
import numpy as np
import gsw as gsw
from scipy import optimize as opt
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy import interpolate as intp
from matplotlib import cm
import matplotlib.colors as colors
from scipy.optimize import linear_sum_assignment
import xarray as xr


def compute_rpe_single_ignore_nan(cost_matrix_2d):
    # Mask invalid entries (NaN values)
    nan_mask = np.isnan(cost_matrix_2d)
    
    # Identify valid rows and columns (at least one non-NaN value)
    valid_rows = ~nan_mask.all(axis=1)
    valid_cols = ~nan_mask.all(axis=0)
    
    # If there are no valid rows or columns, return NaN
    if not valid_rows.any() or not valid_cols.any():
        return np.nan, valid_rows -1, valid_cols-1   ## return -1 might want to write np.nan

    # Extract the valid submatrix
    submatrix = cost_matrix_2d[np.ix_(valid_rows, valid_cols)]

    # Solve the assignment problem on the submatrix
    row_ind, col_ind = linear_sum_assignment(submatrix)
    # Map the submatrix indices back to the original matrix
    original_row_indices = np.where(valid_rows)[0][row_ind]
    original_col_indices = np.where(valid_cols)[0][col_ind]

    # Prepare full-size arrays for indices, padding with -1
    padded_row_ind = np.full(cost_matrix_2d.shape[0], -1)  ## return -1 might want to write np.nan
    padded_col_ind = np.full(cost_matrix_2d.shape[1], -1)  ## return -1 might want to write np.nan

    # Populate the indices with the optimal assignments
    padded_row_ind[original_row_indices] = original_row_indices
    padded_col_ind[original_row_indices] = original_col_indices

    # Sum the costs from the submatrix
    cost_sum = submatrix[row_ind, col_ind].sum() / len(row_ind)
    return cost_sum, padded_row_ind, padded_col_ind


def interpolate_pchip_column(data, z, z_bis):
    """
    Interpolates a single column (Z-dimension) using PchipInterpolator,
    considering only valid (non-NaN) data and masking values outside the valid range.
    """
    # Identify valid data (non-NaN) in the column
    valid_mask = ~np.isnan(data)
    valid_data = data[valid_mask]
    valid_z = z[valid_mask]

    # Only interpolate if there are sufficient valid points
    if len(valid_data) > 1:
        interpolator = PchipInterpolator(valid_z, valid_data)
        interpolated = interpolator(z_bis)
        # Mask interpolated values outside the valid range
        interpolated[z_bis > valid_z[-1]] = np.nan
        return interpolated
    else:
        # If no valid data, return NaNs
        return np.full_like(z_bis, np.nan)
    
def interpolate_xarray(ds,press_bis_broadcasted):
    # Apply the function to the entire dataset
    interpolated = xr.apply_ufunc(
        interpolate_pchip_column,      # Interpolation function
        ds,                            # Original data
        ds['Z'],                       # Original Z coordinates
        press_bis_broadcasted,             # New press_bis coordinates
        input_core_dims=[['Z'], ['Z'], ['press_bis']],  # Core dimensions
        output_core_dims=[['press_bis']],  # Output dimension
        vectorize=True,                # Apply function vectorized
        dask="parallelized",           # Enable Dask if needed
        output_dtypes=[float]          # Output data type
    )
    return interpolated



def mass_of_colunm(P_vec,AS_vec, cT_vec, delta_long, delta_lat, lat):
    rho_vec=gsw.density.rho(AS_vec, cT_vec, P_vec)  
    z=-gsw.z_from_p(P_vec[-1], lat) 
    E_rad=6367.5*10**3 #Earth radius in m
    #assume earth is a perfect sphere 
    vol=z*(delta_long/360)*(delta_lat/360)*(2*np.pi*E_rad)**2 #don't understand the calculation
    col_mass=np.mean(rho_vec)*vol
    return col_mass 
