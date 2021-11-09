from __future__ import division, print_function
import sys
import os
import numpy as np
from astropy.io import fits
from astropy.stats.sigma_clipping import sigma_clip
from mpfit import mpfit
from itertools import product
from tqdm import tqdm

from IPython.core.debugger import Tracer

import matplotlib.pyplot as plt
from matplotlib import rc, patches
rc("font", size=8)
rc("image", interpolation="nearest")
rc("image", origin="lower")
plt.close("all")
plt.ion()

################################################################################
# Options
################################################################################
plot_skyline_fit = False
flux_scale = 1e-17

################################################################################
# File I/O
################################################################################
fnames = sys.argv[1:]
for fname in fnames:
    assert os.path.exists(fname),\
        f"File {fname} not found!"

################################################################################
# Functions used in the mpfit routine
###############################################################################
# The model we fit to the data
def F(lambda_A, p):
    return p[0] * lambda_A + p[1] \
        + p[2] * np.exp(- (lambda_A - p[3])**2 / 2 / p[4]**2)

############################################################################
# Function to be passed into mpfit
def minfunc(p, fjac=None, x=None, y=None, err=None):
    # The model evaluated at the provided x values with parameters stored in p
    model = F(x, p)
    # The deviates are the differences between the measurements y and the model
    # evaluated at each point x normalised by the standard deviation in the
    # measurement.
    deviates = np.array((y - model) / err, dtype=float)
    # We return p status flag and the deviates.
    return 0, deviates


 # Parameters to be passed into the fitting function
parinfo = [{
    'value': 0,
    'fixed': False,
    'limited': [False, False],
    'limits':[0, 0],
    'parname':'',
    'mpside':2,
    'mpprint':0
} for i in range(5)]
parinfo[0]['parname'] = 'm'
parinfo[1]['parname'] = 'b'
parinfo[2]['parname'] = 'A'
parinfo[3]['parname'] = 'mu'
parinfo[4]['parname'] = 'sigma'

# Parameter constraints
# Slope
parinfo[0]['fixed'] = True
# Amplitude: must be positive
parinfo[2]['limited'] = [True, False]
parinfo[2]['limits'] = [0, 0]
# Sigma
parinfo[4]['limited'] = [True, True]
parinfo[4]['limits'] = [0.01, 5.0]
# Mean: constrained to be inside the wavelength window (changes in each iteration)
parinfo[3]['limited'] = [True, True]

################################################################################
# Set up the figure 
################################################################################
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
ax1, ax2 = axs

################################################################################
# Sky subtraction
################################################################################
for fname in fnames:
    print("Now processing sky subtraction for frame {:}...".format(fname))

    # Clear figure
    ax1.clear()
    ax2.clear()

    # Open file
    f = fits.open(fname)
    # Wavelength of the first pixel along the wavelength axis in Angstroms
    lambda_0_A = float(f[0].header.get("CRVAL3"))
    # Sampling of the wavelength axis in Angstroms
    dlambda_A = float(f[0].header.get("CDELT3"))
    N_lambda = float(f[0].header.get("NAXIS3"))     # Number of wavelength values
    lambda_end_A = lambda_0_A + dlambda_A * (N_lambda - 1)
    lambda_vals_A = np.linspace(lambda_0_A, lambda_end_A, N_lambda)

    # Copy data into arrays
    d = np.copy(f[0].data)
    v = np.copy(f[1].data)
    b = np.copy(f[2].data)

    # Mask out bad pixels
    b = b.astype("bool")
    d[b] = np.nan
    v[b] = np.nan

    d_scaled = np.copy(d)
    v_scaled = np.copy(v)

    d_skysub = np.copy(d)
    v_skysub = np.copy(v)

    f.close()

    # List of sky lines to fit to
    skyline_lambdas_A = [l for l in [5577.6, 5934.37, 6364.75, 6499.0, 6605.38, 6950, 6980,
                                     6949.56, 7341.45, 7993.68, 8399.26, 8886.15] if l > lambda_0_A and l < lambda_end_A]

    # Image for selecting sky region
    im = np.nanmedian(d, axis=0)

    fig.suptitle(fname)
    ax1.imshow(im, origin="lower", cmap="afmhot", vmin=np.nanmin(im[2:]), vmax=np.nanmax(im[2:]))
    ax1.set_title("Median slice")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Corner coordinates of sky region
    while True:
        print("Enter coordinates of sky region corners (inclusive):")
        while True:
            coord_list = []
            for coord, maxval in zip(["x_1", "x_2", "y_1", "y_2"], [im.shape[1], im.shape[1], im.shape[0], im.shape[0]]):
                while True:
                    c = input(f"{coord}: ")
                    if c.isdigit() and ~c.isdecimal():
                        if int(c) > maxval - 1:
                            print(f"Error: {coord} must not exceed {maxval - 1}!")
                        else:
                            break
                    else:
                        print(f"Error: {coord} must be a positive integer! Try again...")
                coord_list.append(int(c))
            # Now, check that the coordinates have the right sense 
            x_1, x_2, y_1, y_2 = coord_list
            if x_1 < x_2 and y_1 < y_2:
                break
            else:
                if x_1 >= x_2:
                    print(f"Error: x_1 must be less than x_2! Try again...")
                if y_1 >= y_2:
                    print(f"Error: y_1 must be less than y_2! Try again...")

        # Show the sky region
        # Indicating the selection region of interest on the figure
        rect = patches.Rectangle(
            (x_1 - 0.5, y_1 - 0.5), x_2 - x_1 + 1, y_2 - y_1 + 1,
            linewidth=2, edgecolor="limegreen", facecolor="limegreen", alpha=0.2)
        ax1.add_patch(rect)
        ax2.imshow(im[y_1:y_2 + 1, x_1:x_2 + 1], origin="lower", cmap="afmhot")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title("Sky region")
        fig.canvas.draw()
        plt.show()

        k = input("Press 'r' to re-select sky region, otherwise hit any other key to continue: ")
        if k == "r":
            # Return to outer while loop
            ax2.clear()
            ax1.patches[0].remove()
            fig.canvas.draw()
        else:
            break

    skyline_amplitudes = np.full(
        (len(skyline_lambdas_A), d.shape[1], d.shape[2]), fill_value=np.nan)
    scaling_factors = np.full(
        (len(skyline_lambdas_A), d.shape[1], d.shape[2]), fill_value=np.nan)
    scaling_factor = np.full((d.shape[1], d.shape[2]), fill_value=np.nan)

    # Fit Gaussians to sky lines in a patch of sky
    if plot_skyline_fit:
        fig_fit, ax = plt.subplots(nrows=1, ncols=len(
            skyline_lambdas_A), squeeze=False)

    # Add 1 to the end points so that x_2, y_2 are included
    y_2 += 1
    x_2 += 1

    print("Calculating skyline scaling factors in sky region...")
    for y, x in tqdm(product(range(y_1, y_2), range(x_1, x_2)), total=(y_2 - y_1) * (x_2 - x_1)):
        # Scaling to avoid numerical precision issues
        spectrum = np.copy(d[:, y, x]) / flux_scale
        spectrum_err = np.sqrt(np.copy(v[:, y, x])) / flux_scale

        for s, skyline_lambda_A in enumerate(skyline_lambdas_A):

            sz = 10  # Width of wavelength window
            lambda_obs_idx = int(
                np.round((skyline_lambda_A - lambda_0_A) / dlambda_A))
            spectrum_windowed = spectrum[lambda_obs_idx -
                                         sz // 2:lambda_obs_idx + sz // 2]
            spectrum_windowed -= np.nanmedian(spectrum_windowed)
            spectrum_err_windowed = spectrum_err[lambda_obs_idx -
                                                 sz // 2:lambda_obs_idx + sz // 2]
            lambda_vals_windowed_A = lambda_vals_A[lambda_obs_idx -
                                                   sz // 2:lambda_obs_idx + sz // 2]

            # If there are any NaN pixels then we don't fit, because it may be unreliable.
            # In this case we set the scaling factor to NaN.
            n_bad_pix = len(np.isnan(spectrum_windowed)[
                            np.isnan(spectrum_windowed)])

            if n_bad_pix == 0:
                # Computing the least squares fit
                parnames = {
                    'x': lambda_vals_windowed_A,
                    'y': spectrum_windowed,
                    'err': spectrum_err_windowed
                }
                # Limit the mean
                parinfo[3]['limits'] = [
                    min(lambda_vals_windowed_A), max(lambda_vals_windowed_A)]

                p0 = np.array([0.0, np.nanmedian(spectrum_windowed), np.nanmax(
                    spectrum_windowed), skyline_lambda_A, 0.5])  # p = [slope, intercept, amplitude, mean, stddev]
                fit = mpfit.mpfit(
                    minfunc, p0, functkw=parnames, parinfo=parinfo, quiet=1)
                p_fit = fit.params
                spec_fit = F(lambda_vals_windowed_A, p_fit)

                # Sometimes the fit just fails for no apparent reason - so we assume the scaling = value of peak pixel
                if p_fit[2] == 0:
                    skyline_amplitudes[s, y, x] = np.nanmax(
                        spectrum_windowed)
                else:
                    skyline_amplitudes[s, y, x] = p_fit[2]

                if plot_skyline_fit:
                    ax[0][s].clear()
                    ax[0][s].plot(lambda_vals_windowed_A, (d[lambda_obs_idx - sz // 2:lambda_obs_idx + sz // 2, y_1, x_1] - np.nanmedian(
                        d[lambda_obs_idx - sz // 2:lambda_obs_idx + sz // 2, y_1, x_1])) / flux_scale, label=r"Reference sky spectrum at $(x_1, y_1)$")
                    ax[0][s].plot(lambda_vals_windowed_A, spectrum_windowed, label="Spectrum in this spaxel")
                    ax[0][s].plot(lambda_vals_windowed_A, spec_fit, label="Fit")
                    ax[0][s].set_xlabel(r"Wavelength ($\rm \AA$)")
                    fig_fit.suptitle(f"Spaxel ({x:d}, {y:d})")
                    ax[0][0].legend()
                    fig_fit.canvas.draw()
                    plt.pause(0.5)

        # Calculate scaling parameters to make all of the other lines in other spaxels match the first
        scaling_factors[:, y, x] = skyline_amplitudes[:, y_1, x_1] / skyline_amplitudes[:, y, x]

        # If the fitted amplitude is zero, then the above will be NaN. In this case, do not attempt to calculate the nanmean
        if np.all(np.isnan(scaling_factors[:, y, x])):
            print(f"WARNING: all fitted scaling factors in spaxel ({x:d}, {y:d}) are NaN! Setting scaling_factor in this spaxel to NaN")
            scaling_factor[y, x] = np.nan
        else:
            scaling_factor[y, x] = np.nanmean(sigma_clip(scaling_factors[:, y, x][scaling_factors[:, y, x] > 0], sigma=2))

        # Sometimes the fit to the line is trash even though there aren't any NaN pixels in the spectrum.
        if scaling_factor[y, x] > 5:
            scaling_factor[y, x] = np.nan

        # Multiply the spaxel in the original datacube by an amount to bring
        # it in line with the skyline strength of the master spaxel.
        if np.isnan(scaling_factor[y, x]):
            d_scaled[:, y, x] *= 1.0
            v_scaled[:, y, x] *= 1.0**2
        else:
            d_scaled[:, y, x] *= scaling_factor[y, x]
            v_scaled[:, y, x] *= scaling_factor[y, x]**2

    # Sigma clip mean the spectrum in each spaxel to give the master sky.
    sky_specs = np.copy(d_scaled[:, y_1:y_2, x_1:x_2])
    sky_specs_var = np.copy(v_scaled[:, y_1:y_2, x_1:x_2])

    # Mask out the spaxels with NaN scaling factors. We don't want these spectra contributing to the sky spectrum.
    sky_specs[:, np.isnan(scaling_factor[y_1:y_2, x_1:x_2])] = np.nan
    sky_specs_var[:, np.isnan(scaling_factor[y_1:y_2, x_1:x_2])] = np.nan

    # Reshape to a N_lambda x n_sky_spec array
    n_sky_spec = sky_specs_var.shape[1] * sky_specs_var.shape[2]
    sky_specs = np.reshape(
        sky_specs, (sky_specs.shape[0], n_sky_spec))
    sky_specs_var = np.reshape(
        sky_specs_var, (sky_specs_var.shape[0], n_sky_spec))
    
    # Sigma clip
    sky_specs_clipped = sigma_clip(sky_specs, axis=1)
    sky_specs_var[sky_specs_clipped.mask] = np.nan 
    
    # Evaluate the mean of the scaled sky spectra - this is the master sky spectrum.
    sky_spec = np.nansum(sky_specs_clipped, axis=1) / n_sky_spec
    sky_spec_var = np.nansum(sky_specs_var, axis=1) / n_sky_spec**2

    # Then, for EVERY other spaxel in the image, fit the Gaussians again to
    # sky lines & calculate the multiplicative scale factor required to match
    # the master sky spectrum
    print("Subtracting sky spectrum from all other spaxels...")
    for y, x in tqdm(product(range(d.shape[1]), range(d.shape[2])), total=d.shape[1] * d.shape[2]):
        # print(
            # "Fitting skylines in spaxel y = {0}, x = {1}...".format(y, x))

        # Scaling to avoid numerical precision issues
        spectrum = np.copy(d[:, y, x]) / flux_scale
        spectrum_err = np.sqrt(np.copy(v[:, y, x])) / flux_scale

        for s, skyline_lambda_A in enumerate(skyline_lambdas_A):

            lambda_obs_idx = int(
                np.round((skyline_lambda_A - lambda_0_A) / dlambda_A))
            spectrum_windowed = spectrum[lambda_obs_idx -
                                         sz // 2:lambda_obs_idx + sz // 2]
            spectrum_windowed -= np.nanmedian(spectrum_windowed)
            lambda_vals_windowed_A = lambda_vals_A[lambda_obs_idx -
                                                   sz // 2:lambda_obs_idx + sz // 2]

            # If there are any NaN pixels then we don't fit, because it may be unreliable.
            # In this case we set the scaling factor to NaN.
            n_bad_pix = len(np.isnan(spectrum_windowed)[
                            np.isnan(spectrum_windowed)])

            if n_bad_pix == 0:
                # Computing the least squares fit
                parnames = {
                    'x': lambda_vals_windowed_A,
                    'y': spectrum_windowed,
                    'err': spectrum_err_windowed
                }

                # Limit the mean
                parinfo[3]['limits'] = [
                    min(lambda_vals_windowed_A), max(lambda_vals_windowed_A)]

                p0 = np.array([0.0, np.nanmedian(spectrum_windowed), np.nanmax(
                    spectrum_windowed), skyline_lambda_A, 0.5])  # p = [slope, intercept, amplitude, mean, stddev]
                fit = mpfit.mpfit(
                    minfunc, p0, functkw=parnames, parinfo=parinfo, quiet=1)
                p_fit = fit.params
                spec_fit = F(lambda_vals_windowed_A, p_fit)

                # Sometimes the fit just fails for no apparent reason - so we assume the scaling = value of peak pixel
                if p_fit[2] == 0:
                    skyline_amplitudes[s, y, x] = np.nanmax(
                        spectrum_windowed)
                else:
                    skyline_amplitudes[s, y, x] = p_fit[2]

                if plot_skyline_fit:
                    ax[0][s].clear()
                    ax[0][s].plot(lambda_vals_windowed_A, (d[lambda_obs_idx - sz // 2:lambda_obs_idx + sz // 2, y_1, x_1] - np.nanmedian(
                        d[lambda_obs_idx - sz // 2:lambda_obs_idx + sz // 2, y_1, x_1])) / flux_scale, label=r"Reference sky spectrum at $(x_1, y_1)$")
                    ax[0][s].plot(lambda_vals_windowed_A, spectrum_windowed, label="Spectrum in this spaxel")
                    ax[0][s].plot(lambda_vals_windowed_A, spec_fit, label="Fit")
                    ax[0][s].set_xlabel(r"Wavelength ($\rm \AA$)")
                    fig_fit.suptitle(f"Spaxel ({x:d}, {y:d})")
                    ax[0][0].legend()
                    fig_fit.canvas.draw()
                    plt.show()
                    plt.pause(0.5)

        # Calculate the scaling factor for this spaxel
        scaling_factors[:, y, x] = skyline_amplitudes[:, y_1, x_1] / skyline_amplitudes[:, y, x]
        
        # If the fitted amplitude is zero, then the above will be NaN. In this case, do not attempt to calculate the nanmean
        if np.all(np.isnan(scaling_factors[:, y, x])):
            print(f"WARNING: all fitted scaling factors in spaxel ({x:d}, {y:d}) are NaN! Setting scaling_factor in this spaxel to NaN")
            scaling_factor[y, x] = np.nan
        else:
            scaling_factor[y, x] = np.nanmean(sigma_clip(scaling_factors[:, y, x][scaling_factors[:, y, x] > 0], sigma=2))
        
        # Sometimes the fit to the line is trash even though there aren't any NaN pixels in the spectrum.
        if scaling_factor[y, x] > 5:
            scaling_factor[y, x] = np.nan

        # Subtract the master sky.
        if np.isnan(scaling_factor[y, x]):
            d_skysub[:, y, x] -= sky_spec
            v_skysub[:, y, x] += sky_spec_var
        else:
            d_skysub[:, y, x] -= sky_spec / scaling_factor[y, x]
            v_skysub[:, y, x] += sky_spec_var / scaling_factor[y, x]**2

    # Save sky-corrected frames to file
    f = fits.open(fname)
    f[0].data = d_skysub
    f[1].data = v_skysub
    f[0].header["comment"] = "Sky subtraction performed in python"
    # f.writeto(fname.split('_noskysub.fits')[0] + "_python_skysub.fits", output_verify="ignore", overwrite=True)
    f.writeto(fname.split(".fits")[0] + "_python_skysub.fits", output_verify="ignore", overwrite=True)
    print(f"Saving to file {fname.split('.fits')[0] + '_python_skysub.fits'}...")

print("All done!")
