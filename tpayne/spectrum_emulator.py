from typing import Callable, Generic, List, TypeVar
import numpy as np
from tpayne.utils import classproperty


T = TypeVar("T")


class SpectrumEmulator(Generic[T]):
    @classproperty
    def label_names(cls) -> List[str]:
        return []
    
    @staticmethod
    def is_in_bounds(parameters: T) -> bool:
        return True
    
    @classproperty
    def default_parameters(cls) -> T:
        return np.array([])

    @staticmethod
    def to_parameters() -> T:
        return np.array([])
    
    @classproperty
    def flux_method(cls) -> Callable[..., T]:
        return lambda x: x
