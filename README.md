# scaled_sky_subtraction.py 

### Description
A python routine for using blank regions of an optical wavelength-range data cube to subtract the foreground sky spectrum, which may be useful when (1) the target object does not fill the field-of-view, (2) when the simple sky frame subtraction leaves significant residuals and/or (3) when sky frames are missing.

The routine works as follows. A blank region within the FoV is chosen by the user. Within every spaxel in this region, Gaussian profiles are fitted to a series of strong skylines. Each spaxel in the sky region is scaled by the strength of its skylines before being added together to create a high-S/N "master" sky spectrum. Then, in every other spaxel of the datacube, Gaussian profiles are similarly fitted to the same set of strong skylines. The master sky spectrum is scaled by the strength of these lines before being subtracted from the spaxel, so that minimal residuals are left. The variance of the scaled sky spectrum is also added to the variance data cube. 

This routine has been specifically developed for WiFeS data cubes - the technique should work with optical IFU data from other telescopes, but the FITS format may vary, which may cause runtime errors. 

For details, see Zovaro et al. (2020):
    https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.4940Z/abstract 

### Dependencies
* numpy
* astropy
* mpfit
* tqdm
* itertools
* matplotlib

Tested in python 3.7.

### Basic usage
    python scaled_sky_subtraction.py <fname>.fits 

The sky-subtracted data is saved to the FITS file <fname>_python_skysub.fits in the same directory as the input file.
    
If the "plot_skyline_fit" flag on line 22 is set to True, the Gaussian fits to every skyline are shown. Note that turning this flag on massively slows execution! 

A sample FITS file of the z ~ 0.2 radio galaxy 4C 14.82 has been included as an example.
