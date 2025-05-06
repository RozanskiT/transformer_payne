# Transformer Payne

![Python Version](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue) ![Build workflow](https://github.com/RozanskiT/transformer_payne/actions/workflows/python-package.yml/badge.svg) [![Tests](docs/badges/test.svg)](reports/junit/junit.xml) [![Coverage Status](docs/badges/coverage-badge.svg)](reports/coverage/coverage.xml)

---

Tomasz Różański, Yuan-Sen Ting and Maja Jabłońska, "Toward a Spectral Foundation Model: An Attention-Based Approach with Domain-Inspired Fine-Tuning and Wavelength Parameterization", https://arxiv.org/abs/2306.15703

---

Important note: The implementation of TransformerPayne in this repository includes slight modifications from the version described in the paper above. These changes involve most importantly the placement of residual connections and hyperparameter tuning. Detailed explanations will be provided in an upcoming paper.

## Example usage:
```
import jax
jax.config.update("jax_enable_x64", True)

import transformer_payne as tp
import numpy as np
import matplotlib.pyplot as plt

emulator = tp.TransformerPayne.download() # Download weights for default emulator

wave = np.linspace(4670, 4960, 20000)
mu = 1.0 # Ray perpendicular to solar surface
parameters = emulator.solar_parameters # Pick solar parameters and abundances

spectrum = emulator(np.log10(wave), mu, parameters) # Emulate a spectrum
```

Visulize the results:

```
intensity = spectrum[:,0]
continuum = spectrum[:,1]
normalized_intensity = intensity / continuum

fig, axs = plt.subplots(2, sharex=True, figsize=(5, 4))
axs[0].plot(wave, intensity, color='black', label='Intensity')
axs[0].plot(wave, continuum, color='red', label='Intensity continuum')
axs[1].plot(wave, normalized_intensity, label="Normalized intensity", color='black')
axs[1].vlines(4862.721, 0, 1, color='C0', label=r'$H_\beta$ (vacuum wavelength)')  # Using raw string for LaTeX
axs[1].set_xlabel("Wavelength [$\AA$]")  # Corrected to set xlabel on the second subplot
axs[0].set_ylabel("Intensity [erg/s/cm$^3$/ster]")
axs[1].set_ylabel("Normalized Intensity")
axs[0].legend(loc="lower left")
axs[1].legend(loc="lower left")
plt.subplots_adjust(hspace=0.05)
plt.show()
```
![example_spectrum](docs/readme_plot.png)

## Installation

### Installation using pip

Install transformer-payne along with the `huggingface-hub` and `joblib` modules, which enable easy downloading of model weights from the internet.  
Make sure you also have `matplotlib` and `numpy` installed in order to run the example shown above.  
If you plan to use the notebooks with additional examples located in the `tutorial/` directory, please also install `jupyterlab`, `notebook`, and `ipykernel`.

```
pip install transformer-payne
pip install huggingface-hub joblib numpy matplotlib
pip install jupyterlab notebook ipykernel
```

### Developer installation from local repository

```
git clone https://github.com/RozanskiT/transformer_payne.git
cd transformer_payne
pip install -e .[dev]
```

---

## Available Emulators

### MARCS, Korg.jl and Kurucz's linelist - default intensity emulator

**Disclaimer**: This is an experimental version of the emulator. We do not recommend using it to infer atmospheric parameters of stars, as it currently has significant limitations. However, it is valuable for method development and benchmarking. We are actively working on a future release of emulators with well-understood and validated precision suitable for inference. If you have any questions, concerns, or are interested in developing an emulator for your own spectral grid (in intensity, fluxes, or potentially including Stokes parameters), please don’t hesitate to contact me.

This document outlines an emulator built upon a slightly adapted version of the Korg.jl package and the MARCS stellar atmosphere grid. It simulates stellar spectra across a broad parameter space:

- Effective Temperature (Teff): 4,000 - 8,000 K
- Surface Gravity (logg): 2.0 - 5.0
- Microturbulence (vmic): 0 - 5 km/s
- Metallicity ([Fe/H]): -2.5 to 1.0
- Alpha-element Enhancement ([alpha/Fe], ): -1.0 to 1.0
- Carbon-to-Iron Ratio ([C/Fe]): -1.0 to 1.0
- Elemental Abundances: Individual abundances can vary within a logarithmic range of ±0.3, from Helium to Uranium (with respect to given [Fe/H], [alpha/Fe] and [C/Fe])
- Resolution: ~300,000
- Wavelengths span: 1,500 to 20,000 angstroms

The emulator employs the Kurucz's linelist `gfall08oct17.dat` in Korg.jl code. For more information, visit the following links:

- Korg.jl: [link](https://github.com/ajwheeler/Korg.jl)
- Kurucz's linelist `gfall08oct17.dat`: [link](http://kurucz.harvard.edu/linelists/gfnew/)
- MARCS Grid: [link](https://dr17.sdss.org/sas/dr17/apogee/spectro/speclib/atmos/marcs/MARCS_v3_2016/Readme_MARCS_v3_2016.txt)

### Work in progress...
