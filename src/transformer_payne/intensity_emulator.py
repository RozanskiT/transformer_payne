from abc import abstractmethod
from transformer_payne.spectrum_emulator import SpectrumEmulator
from typing import TypeVar


T = TypeVar("T")

class IntensityEmulator(SpectrumEmulator[T]):
    @abstractmethod
    def stellar_parameter_names(self) -> T:
        """Get labels of stellar parameters (no geometry-related parameters, e.g. mu)

        Returns:
            T:
        """
        raise NotImplementedError
    
    @abstractmethod
    def min_stellar_parameters(self) -> T:
        """Get minimum values of stellar parameters (no geometry-related parameters, e.g. mu)

        Returns:
            T:
        """
        raise NotImplementedError
    
    @abstractmethod
    def max_stellar_parameters(self) -> T:
        """Get minimum values of stellar parameters (no geometry-related parameters, e.g. mu)

        Returns:
            T:
        """
        raise NotImplementedError
    
    @abstractmethod
    def intensity(self, log_wavelengths: T, mu: float, parameters: T) -> T:
        """Calculate the intensity for given wavelengths and mus

        Args:
            log_wavelengths (T): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight
            spectral_parameters (T): an array of predefined stellar parameters

        Returns:
            T: intensities corresponding to passed wavelengths [erg/cm2/s/angstrom]
        """
        raise NotImplementedError
