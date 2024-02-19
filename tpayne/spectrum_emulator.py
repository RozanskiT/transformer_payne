from typing import Callable, Generic, List, TypeVar
import numpy as np
from tpayne.utils import classproperty


T = TypeVar("T")


class SpectrumEmulator(Generic[T]):
    @classproperty
    def label_names(cls) -> List[str]:
        return []
    
    @classproperty
    def min_parameters(cls) -> T:
        return np.array([])
    
    @classproperty
    def max_parameters(cls) -> T:
        return np.array([])
    
    @classmethod
    def is_in_bounds(cls, parameters: T) -> bool:
        return np.all(parameters >= cls.min_parameters) and np.all(parameters <= cls.max_parameters)
    
    @classproperty
    def solar_parameters(cls) -> T:
        return np.array([])

    @classmethod
    def to_parameters(cls) -> T:
        return np.array([])
    
    @classproperty
    def flux_method(cls) -> Callable[..., T]:
        return lambda x: x
