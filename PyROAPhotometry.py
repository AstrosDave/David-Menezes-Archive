#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyROA lag analysis for rm101
Generates g-band and Hβ light curves with different filter widths,
then uses PyROA to measure lags.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from scipy import integrate
import PyROA as roa   # <-- PyROA

# ============================================================
# Read spectrum
# ============================================================
def read_spectrum(file):
    with fits.open(file) as hdu:        data1 = hdu[1].data
        data2 = hdu[2].data

        IVAR = data1["IVAR"]
        flux = data1["FLUX"]
        mjd = data2['MJD']
        wav = 10**(data1["LOGLAM"])
        z = data2["z"][0]
        restwavelength = wav / (1 + z)

        # Flux mask
        flux_mask = np.isfinite(flux) & (flux > -1e5) & (flux < 1e5)

        # NMAD uncertainty
        def nmad(arr):
            return 1.4826 * np.median(np.abs(arr - np.median(arr)))
        flux_nmad = nmad(flux[flux_mask]) if np.any(flux_mask) else np.nan
        uncertainty = np.full_like(flux, flux_nmad)

        mask = flux_mask & np.isfinite(uncertainty) & (IVAR > 0)

        return wav[mask], flux[mask], uncertainty[mask], restwavelength[mask], mjd, z

# ============================================================
# Filters
# ============================================================
def hbeta_filter(wavelength, width=1370):
    center = 4861
    return (wavelength >= (center - width/2)) & (wavelength <= (center + width/2))

def g_filter(wavelength):
    # Rough Sloan g-band range
    return (wavelength >= 4000) & (wavelength <= 5500)

# ============================================================
# Monte Carlo photometry
# ============================================================
def monte_carlo_photometry(wavelength, flux, filter_mask, uncertainty, n_iterations=20):
    fluxes = []
    for _ in range(n_iterations):
        perturbed_flux = np.random.normal(flux, uncertainty)
        flux_in_filter = perturbed_flux[filter_mask]
        fluxes.append(integrate.simpson(flux_in_filter, x=wavelength[filter_mask]))
    return fluxes

# ============================================================
# Main script
# ============================================================
if __name__ == "__main__":
    direc = "/Users/davidmenezes/Downloads/2023-24 Research/rm101/Main/"

    widths = [1370, 1000, 500, 200, 100]   # Hβ filter widths to test
    lightcurves = {}  # dict: width -> (mjd, flux, err)
    g_fluxes, g_errs, g_mjd = [], [], []

    # Loop through spectra
    for ep in os.listdir(direc):
        file_path = os.path.join(direc, ep)
        if os.path.isfile(file_path):
            wav, flux, uncertainty, restwavelength, mjd, z = read_spectrum(file_path)

            # g-band
            gmask = g_filter(wav)
            gflux_mc = monte_carlo_photometry(wav, flux, gmask, uncertainty)
            g_fluxes.append(np.median(gflux_mc))
            g_errs.append(np.std(gflux_mc))
            g_mjd.append(mjd)

            # Hβ with different widths
            for w in widths:
                filt = hbeta_filter(wav, width=w)
                fluxes_mc = monte_carlo_photometry(wav, flux, filt, uncertainty)
                fluxes_mc = np.asarray(fluxes_mc) / w
                if w not in lightcurves:
                    lightcurves[w] = ([], [], [])
                lightcurves[w][0].append(mjd)
                lightcurves[w][1].append(np.median(fluxes_mc))
                lightcurves[w][2].append(np.std(fluxes_mc))

    # Convert to PyROA LightCurve objects
    g_lc = roa.LightCurve(np.array(g_mjd), np.array(g_fluxes), np.array(g_errs))
    hbeta_lcs = {w: roa.LightCurve(np.array(mjd), np.array(flux), np.array(err))
                 for w, (mjd, flux, err) in lightcurves.items()}

    # Run PyROA for each Hβ width
    lag_results = {}
    for w, lc in hbeta_lcs.items():
        print(f"\n=== Running PyROA for Hβ width={w} Å ===")
        run = roa.Run([g_lc, lc])
        run.set_priors(lag_bounds=[0, 200], width_bounds=[0.1, 100])
        run.do_run(nwalkers=50, nsteps=2000, burn=500)

        # Get lag samples
        lag_samples = run.chain["lag"][:, 1]  # 2nd LC
        lag_median = np.median(lag_samples)
        lag_err = np.std(lag_samples)
        lag_results[w] = (lag_median, lag_err)
        print(f"Hβ width={w} Å → Lag = {lag_median:.2f} ± {lag_err:.2f} days")

    # Plot light curves
    plt.figure(figsize=(12, 6))
    for w, (mjd, flux, err) in lightcurves.items():
        plt.errorbar(mjd, flux, yerr=err, fmt="o", label=f"Hβ {w} Å")
    plt.errorbar(g_mjd, g_fluxes, yerr=g_errs, fmt="s", label="g-band", color="black")
    plt.xlabel("MJD")
    plt.ylabel("Flux")
    plt.title("g-band and Hβ Light Curves")
    plt.legend()
    plt.show()

    # Plot lag vs filter width
    widths_plot = sorted(lag_results.keys())
    lags = [lag_results[w][0] for w in widths_plot]
    errors = [lag_results[w][1] for w in widths_plot]

    plt.errorbar(widths_plot, lags, yerr=errors, fmt="o-")
    plt.xlabel("Hβ Filter Width (Å)")
    plt.ylabel("Lag (days)")
    plt.title("Lag vs Hβ Filter Width (PyROA)")
    plt.show()
