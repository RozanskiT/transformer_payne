from typing import Any, Dict, List

from transformer_payne.spectrum_emulator import SpectrumEmulator

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


class Blackbody(SpectrumEmulator[ArrayLike]):
    @property
    def parameter_names(self) -> List[str]:
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
    
    @property
    def stellar_parameter_names(self) -> ArrayLike:
        return self.parameter_names()
    
    @property
    def min_stellar_parameters(self) -> ArrayLike:
        return self.min_stellar_parameters()
    
    @property
    def max_stellar_parameters(self) -> ArrayLike:
        return self.max_stellar_parameters()
    
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
    
    def to_parameters(self, parameter_values: Dict[str, Any] = None) -> ArrayLike:
        """Convert passed values to the accepted parameters format

        Args:
            parameter_values (Dict[str, Any], optional): parameter values in the format of {'parameter_name': value}. Unset parameters will be set to solar values.

        Raises:
            ValueError: when the parameters are out of accepted bounds

        Returns:
            ArrayLike:
        """
        if not parameter_values:
            return self.solar_parameters
        
        teff = parameter_values.get('teff', self.solar_parameters[0])
        if teff<0.0:
            raise ValueError("Effective temperature is out of bounds! It must be non-negative")
        
        return jnp.array([teff])
    
    @staticmethod
    def flux(log_wavelengths: ArrayLike, spectral_parameters: ArrayLike) -> ArrayLike:
        """Compute the blackbody flux.

        Args:
            log_wavelengths (ArrayLike): Array of logarithmic wavelengths (log10 of wavelength in Angstroms).
            parameters (ArrayLike): Array of parameters. In this case, only one element is used which represents the temperature in Kelvin.

        Returns:
            ArrayLike: Array of blackbody monochromatic fluxes in erg/s/cm3
        """
        
        return jnp.pi * Blackbody.intensity(log_wavelengths, None, spectral_parameters)

    @staticmethod
    def intensity(log_wavelengths: ArrayLike, mu: float, spectral_parameters: ArrayLike) -> ArrayLike:
        """Calculate the intensity for given wavelengths and mus

        Args:
            log_wavelengths (ArrayLike): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight. As the blackbody radiation field is isotropic, this parameter is not used.
            spectral_parameters (ArrayLike): an array of predefined stellar parameters. In this case, only one element is used which represents the temperature in Kelvin.

        Returns:
            ArrayLike: monochromatic intensities corresponding to passed wavelengths [erg/s/cm3/steradian]
        """
        # Convert log wavelength from angstroms to cm
        wave_cm = 10 ** (log_wavelengths - 8)  # 1 Angstrom = 1e-8 cm

        # Extract temperature from parameters
        spectral_parameters = jnp.atleast_1d(spectral_parameters)
        T = spectral_parameters[0]

        # Compute blackbody intensity
        intensity = ((2 * h * c ** 2 / wave_cm ** 5 * 1 / (jnp.exp(h * c / (wave_cm * k * T)) - 1)))

        return intensity[:, jnp.newaxis].repeat(2, axis=1)
