from abc import ABC, abstractmethod, abstractproperty
from typing import Generic, List, TypeVar
import numpy as np


T = TypeVar("T")


class SpectrumEmulator(ABC, Generic[T]):
    @abstractproperty
    def label_names(self) -> List[str]:
        """Get labels of spectrum model parameters

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
    def to_parameters(self) -> T:
        """Convert passed values to the accepted parameters format

        Returns:
            T:
        """
        raise NotImplementedError
