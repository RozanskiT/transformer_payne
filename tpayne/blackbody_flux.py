from typing import Callable, Dict, List, Union
from tpayne.flux_emulator import FluxEmulator
import tpayne.tpayne_consts as const

SOLAR_TEFF = 5777 # K

try:
    import jax.numpy as jnp
    from jax.typing import ArrayLike
    
except ImportError:
    import numpy as jnp
    from numpy.typing import ArrayLike
    
h = 6.62607015e-27  # Planck's constant [erg*s]
c = 2.99792458e10   # Speed of light [cm/s]
k = 1.380649e-16    # Boltzmann constant [erg/K]


class BlackbodyFlux(FluxEmulator[ArrayLike]):
    @property
    def label_names(self) -> List[str]:
        """Get labels of spectrum model parameters

        Returns:
            List[str]:
        """
        return ["teff"]
    
    @property
    def min_parameters(self) -> ArrayLike:
        """Minimum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return jnp.array([0.], dtype=jnp.float32)
    
    @property
    def max_parameters(self) -> ArrayLike:
        """Maximum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return jnp.array([jnp.inf], dtype=jnp.float32)
    
    @staticmethod
    def is_in_bounds(parameters: ArrayLike) -> bool:
        """Check if parameters are within the bounds of the spectrum model

        Args:
            parameters (ArrayLike):

        Returns:
            bool:
        """
        return jnp.all(parameters >= 0.)
    
    @property
    def solar_parameters(self) -> ArrayLike:
        """Solar parameters for the spectrum model

        Returns:
            ArrayLike:
        """
        return jnp.array([5777])
    
    # zrobic zeby sie automatycznie generowaly slowka kluczowe takie jak mamy w labelach
    @staticmethod
    def to_parameters(teff: float = 5777.0) -> ArrayLike:
        """Convert passed values to the accepted parameters format

        Args:
            teff (float, optional): effective temperature [K]. Defaults to solar teff=5777 K.

        Raises:
            ValueError: when the parameters are out of accepted bounds

        Returns:
            ArrayLike:
        """
        if teff<0.0:
            raise ValueError("Effective temperature is out of bounds! It must be non-negative")
        
        return jnp.array([teff])
    
    @staticmethod
    def flux(log_wavelengths: ArrayLike, parameters: ArrayLike) -> ArrayLike:
        """Compute the blackbody flux.

        Args:
            log_wavelengths (ArrayLike): Array of logarithmic wavelengths (log10 of wavelength in Angstroms).
            parameters (ArrayLike): Array of parameters. In this case, only one element is used which represents the temperature in Kelvin.

        Returns:
            ArrayLike: Array of blackbody fluxes in erg/s/cm2/A
        """
        # Convert log wavelength from angstroms to cm
        wave_cm = 10 ** (log_wavelengths - 8)  # 1 Angstrom = 1e-8 cm

        # Extract temperature from parameters
        T = parameters[0]

        # Compute blackbody intensity
        intensity = ((2 * h * c ** 2 / wave_cm ** 5 * 1 / (jnp.exp(h * c / (wave_cm * k * T)) - 1)))*1e-8

        return intensity
