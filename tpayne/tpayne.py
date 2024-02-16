from typing import Callable, List
from tpayne.exceptions import JAXWarning
from tpayne.spectrum_emulator import SpectrumEmulator
from tpayne.utils import classproperty
import warnings

try:
    import jax.numpy as jnp
    from jax.typing import ArrayLike
except ImportError:
    import numpy as jnp
    import numpy.typing as ArrayLike
    warnings.warn("Please install JAX to use TPayne.", JAXWarning)


class TPayne(SpectrumEmulator[ArrayLike]):
    @classproperty
    def label_names(cls) -> List[str]:
        return []
    
    @staticmethod
    def is_in_bounds(parameters: ArrayLike) -> bool:
        return True
    
    @classproperty
    def default_parameters(cls) -> ArrayLike:
        return jnp.array([])
    
    @staticmethod
    def to_parameters() -> ArrayLike:
        return jnp.array([])
    
    @classproperty
    def flux_method(cls) -> Callable[..., ArrayLike]:
        return lambda x: x
