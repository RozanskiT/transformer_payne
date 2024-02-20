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
    
    def is_in_bounds(self, parameters: ArrayLike) -> bool:
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
    @classmethod
    def to_parameters(self,
                      teff: float = 5777.0,
                      logg: float = 4.44,
                      mu: float = 1.0,
                      abundances: Union[ArrayLike, Dict[str, float]] = const.SOLAR_ABUNDANCES) -> ArrayLike:
        """Convert passed values to the accepted parameters format

        Args:
            logteff (float, optional): log10(effective temperature) [log(K)]. Defaults to solar log(teff)=3.7617023675414125.
            logg (float, optional): log10(g) [log(m/s2)]. Defaults to solar logg=4.44.
            mu (float, optional): [0.0-1.0]. Defaults to 1.0.
            abundances (Union[ArrayLike, Dict[str, float]], optional): abundances relative to the solar abundance 
                passed either as an array or dict in the form of {'element name': 0.0} (refer to label names).
                Defaults to solar abundances (all elements=0.0).

        Raises:
            ValueError: when the parameters are out of accepted bounds

        Returns:
            ArrayLike:
        """
        
        if isinstance(abundances, dict):
            abundance_values = jnp.array([abundances.get(element, 0.) for element in const.ELEMENTS])
        else:
            abundance_values = abundances

        parameters = jnp.concatenate([jnp.array([teff, logg, mu]), abundance_values])
        if not self.is_in_bounds(parameters):
            raise ValueError("Parameters are not within accepted bounds. Refer to tpayne.tpayne_consts for accepted bounds.")
        
        return parameters
    
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

        return jnp.tile(intensity, (2, 1))
