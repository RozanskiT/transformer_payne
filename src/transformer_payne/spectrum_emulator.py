from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, Generic, List, TypeVar
import numpy as np


T = TypeVar("T")


class SpectrumEmulator(ABC, Generic[T]):
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

    @abstractproperty
    def parameter_names(self) -> List[str]:
        """Get labels of model parameters

        Returns:
            List[str]:
        """
        raise NotImplementedError
    
    @abstractproperty
    def min_parameters(self) -> T:
        """Minimum values supported by the spectrum model

        Returns:
            T:
        """
        raise NotImplementedError
    
    @abstractproperty
    def max_parameters(self) -> T:
        """Maximum values supported by the spectrum model

        Returns:
            T:
        """
        raise NotImplementedError
    
    def is_in_bounds(self, parameters: T) -> bool:
        """Check if parameters are within the bounds of the spectrum model

        Args:
            parameters (T):

        Returns:
            bool:
        """
        return np.all(parameters >= self.min_parameters) and \
            np.all(parameters <= self.max_parameters)
    
    @abstractproperty
    def solar_parameters(self) -> T:
        """Solar parameters for the spectrum model

        Returns:
            T:
        """
        raise NotImplementedError

    @abstractmethod
    def to_parameters(self, parameter_values: Dict[str, Any] = None) -> T:
        """Convert passed values to the accepted parameters format

        Args:
            parameter_values (Dict[str, Any], optional): parameter values in the format of {'parameter_name': value}. Unset parameters will be set to solar values.

        Returns:
            T:
        """
        raise NotImplementedError

    def print_parameter_bounds(self):
        print("Parameter bounds:")
        print("NAME\tMIN\tMAX")
        for param_name, (p_min, p_max) in zip(self.parameter_names, zip(self.min_parameters, self.max_parameters)):
            print(f"{param_name}\t{p_min:.4f}\t{p_max:.4f}")
            
    @abstractmethod
    def flux(self, log_wavelengths: T, parameters: T) -> T:
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
