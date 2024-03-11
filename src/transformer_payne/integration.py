from typing import Callable
import numpy as np
import warnings

try:
    from jax.typing import ArrayLike
    
except ImportError:
    from numpy.typing import ArrayLike

    
def integrate_intensity(intensity_fn: Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike],
                        mus_number: int = 10) -> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    roots, weights = np.polynomial.legendre.leggauss(mus_number)
    roots = (roots + 1) / 2
    weights /= 2
    
    try:
        import jax
        import jax.numpy as jnp
        roots, weights = jnp.array(roots), jnp.array(weights)
        vec_intensity = jax.vmap(intensity_fn, in_axes=(None, 0, None))
        integrated_func = lambda wavelengths, parameters: 2*jnp.pi*jax.numpy.sum(
            vec_intensity(wavelengths, roots, parameters)*roots[:, jnp.newaxis, jnp.newaxis]*weights[:, jnp.newaxis, jnp.newaxis],
            axis=0
        )
        return jax.jit(integrated_func)
    except ImportError:
        warnings.warn("No JAX installed. Performance can't be optimized.")
        return lambda wavelengths, parameters: 2*np.pi*np.sum([
            intensity_fn(wavelengths, mu, parameters)*mu*w for mu, w in zip(roots, weights)
        ], axis=0)
