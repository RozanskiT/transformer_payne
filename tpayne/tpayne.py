from typing import Callable, Dict, List, Union
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
        """Get labels of spectrum model parameters

        Returns:
            List[str]:
        """
        return const.LABEL_NAMES
    
    @classproperty
    def min_parameters(cls) -> ArrayLike:
        """Minimum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return MIN_PARAMS
    
    @classproperty
    def max_parameters(cls) -> ArrayLike:
        """Maximum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return MAX_PARAMS
    
    @classmethod
    def is_in_bounds(cls, parameters: ArrayLike) -> bool:
        """Check if parameters are within the bounds of the spectrum model

        Args:
            parameters (ArrayLike):

        Returns:
            bool:
        """
        return jnp.all(parameters >= const.MIN_PARAMS) and jnp.all(parameters <= const.MAX_PARAMS)
    
    @classproperty
    def solar_parameters(cls) -> ArrayLike:
        """Solar parameters for the spectrum model

        Returns:
            ArrayLike:
        """
        return SOLAR_PARAMS
    
    @classmethod
    def to_parameters(cls,
                      logteff: float = 3.7617023675414125,
                      logg: float = 4.44,
                      mu: float = 1.0,
                      abundances: Union[ArrayLike, Dict[str, float]] = const.SOLAR_ABUNDANCES) -> ArrayLike:
        """Convert passed values to the accepted parameters format

        Args:
            logteff (float, optional): log10(effective temperature) [log(K)]. Defaults to solar log(teff)=3.7617023675414125.
            logg (float, optional): log10(g) [log(m/s2)]. Defaults to solar logg=4.44.
            mu (float, optional): [0.0-1.0]. Defaults to 1.0.
            abundances (Union[ArrayLike, Dict[str, float]], optional): abundances relative to the solar abundance 
                passed either as an array or dict in the form of {'element name': 0.0} (refer to label names).
                Defaults to solar abundances (all elements=0.0).

        Raises:
            ValueError: when the parameters are out of accepted bounds

        Returns:
            ArrayLike:
        """
        
        if isinstance(abundances, dict):
            abundance_values = jnp.array([abundances.get(element, 0.) for element in const.ELEMENTS])
        else:
            abundance_values = abundances

        parameters = jnp.concatenate([jnp.array([logteff, logg, mu]), abundance_values])
        if not cls.is_in_bounds(parameters):
            raise ValueError("Parameters are not wihin accepted bounds. Refer to tpayne.tpayne_consts for accepted bounds.")
        
        return parameters
    
    @classproperty
    def flux_method(cls) -> Callable[..., ArrayLike]:
        return lambda x: x
