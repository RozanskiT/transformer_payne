from typing import Callable, List
from tpayne.exceptions import JAXWarning
from tpayne.spectrum_emulator import SpectrumEmulator
import tpayne.tpayne_consts as const
from tpayne.utils import classproperty
import warnings

try:
    import jax.numpy as jnp
    from jax.typing import ArrayLike
    
    MAX_PARAMS = jnp.array(const.MAX_PARAMS)
    MIN_PARAMS = jnp.array(const.MIN_PARAMS)
    SOLAR_PARAMS = jnp.array(const.SOLAR_PARAMS)
    
except ImportError:
    import numpy as jnp
    from numpy.typing import ArrayLike
    warnings.warn("Please install JAX to use TPayne.", JAXWarning)
    
    MAX_PARAMS = const.MAX_PARAMS
    MIN_PARAMS = const.MIN_PARAMS
    SOLAR_PARAMS = const.SOLAR_PARAMS


class TPayne(SpectrumEmulator[ArrayLike]):
    @classproperty
    def label_names(cls) -> List[str]:
        return const.LABEL_NAMES
    
    @classproperty
    def min_parameters(cls) -> ArrayLike:
        return MIN_PARAMS
    
    @classproperty
    def max_parameters(cls) -> ArrayLike:
        return MAX_PARAMS
    
    @classmethod
    def is_in_bounds(cls, parameters: ArrayLike) -> bool:
        return jnp.all(parameters >= const.MIN_PARAMS) and jnp.all(parameters <= const.MAX_PARAMS)
    
    @classproperty
    def solar_parameters(cls) -> ArrayLike:
        return SOLAR_PARAMS
    
    @classmethod
    def to_parameters(cls) -> ArrayLike:
        return jnp.array([])
    
    @classproperty
    def flux_method(cls) -> Callable[..., ArrayLike]:
        return lambda x: x
