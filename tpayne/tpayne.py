from typing import Callable, List
from tpayne.exceptions import JAXWarning
from tpayne.spectrum_emulator import SpectrumEmulator
import tpayne.tpayne_consts as const
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
    
    @classproperty
    def min_parameters(cls) -> ArrayLike:
        return const.MIN_PARAMS
    
    @classproperty
    def max_parameters(cls) -> ArrayLike:
        return const.MAX_PARAMS
    
    @classmethod
    def is_in_bounds(cls, parameters: ArrayLike) -> bool:
        return jnp.all(parameters >= const.MIN_PARAMS) and jnp.all(parameters <= const.MAX_PARAMS)
    
    @classproperty
    def default_parameters(cls) -> ArrayLike:
        return jnp.array([])
    
    @classmethod
    def to_parameters(cls) -> ArrayLike:
        return jnp.array([])
    
    @classproperty
    def flux_method(cls) -> Callable[..., ArrayLike]:
        return lambda x: x
