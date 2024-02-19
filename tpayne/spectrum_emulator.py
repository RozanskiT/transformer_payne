from typing import Callable, Generic, List, TypeVar
import numpy as np
from tpayne.utils import classproperty


T = TypeVar("T")


class SpectrumEmulator(Generic[T]):
    @classproperty
    def label_names(cls) -> List[str]:
        """Get labels of spectrum model parameters

        Returns:
            List[str]:
        """
        return []
    
    @classproperty
    def min_parameters(cls) -> T:
        """Minimum values supported by the spectrum model

        Returns:
            T:
        """
        return np.array([])
    
    @classproperty
    def max_parameters(cls) -> T:
        """Maximum values supported by the spectrum model

        Returns:
            T:
        """
        return np.array([])
    
    @classmethod
    def is_in_bounds(cls, parameters: T) -> bool:
        """Check if parameters are within the bounds of the spectrum model

        Args:
            parameters (T):

        Returns:
            bool:
        """
        return np.all(parameters >= cls.min_parameters) and np.all(parameters <= cls.max_parameters)
    
    @classproperty
    def solar_parameters(cls) -> T:
        """Solar parameters for the spectrum model

        Returns:
            T:
        """
        return np.array([])

    @classmethod
    def to_parameters(cls) -> T:
        """Convert passed values to the accepted parameters format

        Returns:
            T:
        """
        return np.array([])
    
    @classproperty
    def flux_method(cls) -> Callable[..., T]:
        """Get method for simulating flux for passed parameters

        Returns:
            Callable[..., T]:
        """
        return lambda x: x
