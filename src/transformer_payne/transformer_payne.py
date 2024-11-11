from typing import Any, Dict, List, Union, Tuple, Callable
from transformer_payne.architecture_definition import ArchitectureDefinition
from transformer_payne.download import download_hf_model
from transformer_payne.exceptions import JAXWarning
from transformer_payne.spectrum_emulator import SpectrumEmulator
from transformer_payne.configuration import REPOSITORY_ID_KEY, FILENAME_KEY
from functools import partial
import warnings
import os
import numpy as np

HF_CONFIG_NAME = "TPAYNE"
ARCHITECTURE_NAME = "TransformerPayneIntensities"

try:
    import jax.numpy as jnp
    from jax.typing import ArrayLike
    import jax
    from flax import linen as nn
    from flax.linen import initializers as init
    from flax.core.frozen_dict import freeze
    
except ImportError:
    import numpy as jnp
    from numpy.typing import ArrayLike
    warnings.warn("Please install JAX and Flax to use TransformerPayne.", JAXWarning)

_activation_functions_dict = {
    'gelu': nn.gelu,
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'linear': lambda x: x
}

def frequency_encoding(x, min_period, max_period, dimension, dtype=jnp.float64):
    periods = jnp.logspace(jnp.log10(min_period), jnp.log10(max_period), num=dimension, dtype=dtype)
    
    y = jnp.sin(2*jnp.pi/periods*x)
    return y

class ParametersEmbedding(nn.Module):
    dim: int = 256
    input_dim: int = 95
    no_tokens: int = 16
    use_bias: bool = False
    activation_fn: Callable = nn.gelu
    init_type: str = "si"  # "si" or "zero"
    sigma: float = 1.0
    alpha: float = 1.0

    @nn.compact
    def __call__(self, x, train):
        dtype = x.dtype
        input_dim = self.input_dim

        # Choose initializer with axis adjustment
        stddev_we0 = input_dim ** -0.5
        stddev_we1 = self.dim ** -0.5
        init_we0 = nn.initializers.truncated_normal(stddev = self.sigma * stddev_we0 / jnp.array(.87962566103423978, dtype))
        init_we1 = {
            "si": nn.initializers.truncated_normal(stddev = self.sigma * stddev_we1 / jnp.array(.87962566103423978, dtype)),
            "zero": init.zeros,
        }[self.init_type]
        b_init = init.zeros

        # Define weight matrices
        w0 = self.param('we0', init_we0, (input_dim, self.dim), dtype=dtype)
        w1 = self.param('we1', init_we1, (self.dim, self.no_tokens * self.dim), dtype=dtype)

        if self.use_bias:
            b0 = self.param('be0', b_init, (self.dim,), dtype=dtype)
            b1 = self.param('be1', b_init, (self.no_tokens * self.dim,), dtype=dtype)

        # First layer computation with jnp.einsum
        p = jnp.einsum('...i,ij->...j', x, w0)
        if self.use_bias:
            p += b0
        p = self.activation_fn(p)

        # Second layer computation with jnp.einsum
        p = jnp.einsum('...i,ij->...j', p, w1)
        if self.use_bias:
            p += b1

        # Reshape output
        p = jnp.reshape(p, (self.no_tokens, self.dim))
        p = self.alpha * p
        return p

class FeedForward(nn.Module):
    dim: int = 256
    dim_ff_multiplier: int = 4
    use_bias: bool = False
    activation_fn: Callable = nn.gelu
    init_type: str = "si"  # "si" or "vs"
    sigma: float = 1.0

    @nn.compact
    def __call__(self, x, train):
        dtype = x.dtype
        input_dim = x.shape[-1]

        # Choose initializer with axis adjustment
        stddev_wfi = input_dim ** -0.5
        stddev_wfo = (self.dim * self.dim_ff_multiplier) ** -0.5
        init_wfi = nn.initializers.truncated_normal(stddev = self.sigma * stddev_wfi / jnp.array(.87962566103423978, dtype))
        init_wfo = {
            "si": nn.initializers.truncated_normal(stddev = self.sigma * stddev_wfo / jnp.array(.87962566103423978, dtype)),
            "zero": init.zeros,
        }[self.init_type]
        b_init = init.zeros

        # Define weight matrices
        w0 = self.param('wfi', init_wfi, (input_dim, self.dim * self.dim_ff_multiplier), dtype=dtype)
        w1 = self.param('wfo', init_wfo, (self.dim * self.dim_ff_multiplier, self.dim), dtype=dtype)

        if self.use_bias:
            b0 = self.param('bfi', b_init, (self.dim * self.dim_ff_multiplier,), dtype=dtype)
            b1 = self.param('bfo', b_init, (self.dim,), dtype=dtype)

        # First layer computation with jnp.einsum
        x = jnp.einsum('...i,ij->...j', x, w0)
        if self.use_bias:
            x += b0
        x = self.activation_fn(x)

        # Second layer computation with jnp.einsum
        x = jnp.einsum('...i,ij->...j', x, w1)
        if self.use_bias:
            x += b1
        return x

class PredictionHead(nn.Module):
    dim: int = 256
    out_dim: int = 1
    use_bias: bool = False
    activation_fn: Callable = nn.gelu
    output_activation_fn: Callable = lambda x: x
    init_type: str = "si"  # "si" or "zero"
    sigma: float = 1.0

    @nn.compact
    def __call__(self, x, train):
        dtype = x.dtype
        input_dim = x.shape[-1]

        # Choose initializer with axis adjustment
        stddev_wp0 = input_dim ** -0.5
        stddev_wp1 = self.dim ** -0.5
        init_wp0 = nn.initializers.truncated_normal(stddev = self.sigma * stddev_wp0 / jnp.array(.87962566103423978, dtype))
        init_wp1 = {
            "si": nn.initializers.truncated_normal(stddev = self.sigma * stddev_wp1 / jnp.array(.87962566103423978, dtype)),
            "zero": init.zeros,
        }[self.init_type]
        b_init = init.zeros

        # Define weight matrices
        w0 = self.param('wp0', init_wp0, (input_dim, self.dim), dtype=dtype)
        w1 = self.param('wp1', init_wp1, (self.dim, self.out_dim), dtype=dtype)

        if self.use_bias:
            b0 = self.param('bp0', b_init, (self.dim,), dtype=dtype)
            b1 = self.param('bp1', b_init, (self.out_dim,), dtype=dtype)

        # First layer computation with jnp.einsum
        x = jnp.einsum('...i,ij->...j', x, w0)
        if self.use_bias:
            x += b0
        x = self.activation_fn(x)

        # Second layer computation with jnp.einsum
        x = jnp.einsum('...i,ij->...j', x, w1)
        if self.use_bias:
            x += b1

        # Apply output activation function
        output = self.output_activation_fn(x)
        return output

class MHA(nn.Module):
    dim: int = 256
    dim_head: int = 32
    use_bias: bool = False
    init_att_q: str = "si"  # "si" or "zero"
    init_att_o: str = "si"  # "si" or "zero"
    sigma: float = 1.0
    alpha_att: float = 1.0

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, train=True):
        num_heads = self.dim // self.dim_head
        dtype = inputs_q.dtype
        
        # Choose initializer
        stddev = self.dim ** -0.5
        init_q = {
            "si": nn.initializers.truncated_normal(stddev = self.sigma * stddev / jnp.array(.87962566103423978, dtype)), 
            "zero": init.zeros,
        }[self.init_att_q]
        init_kv = nn.initializers.truncated_normal(stddev = self.sigma * stddev / jnp.array(.87962566103423978, dtype))
        init_o = {
            "si": nn.initializers.truncated_normal(stddev = self.sigma * stddev / jnp.array(.87962566103423978, dtype)),
            "zero": init.zeros,
        }[self.init_att_o]
        b_init = init.zeros

        # Define weight matrices with shapes that include num_heads and dim_head
        w_q = self.param('w_q', init_q, (self.dim, num_heads, self.dim_head), dtype=dtype)
        w_k = self.param('w_k', init_kv, (self.dim, num_heads, self.dim_head), dtype=dtype)
        w_v = self.param('w_v', init_kv, (self.dim, num_heads, self.dim_head), dtype=dtype)
        w_o = self.param('w_o', init_o, (num_heads, self.dim_head, self.dim), dtype=dtype)

        if self.use_bias:
            b_q = self.param('b_q', b_init, (num_heads, self.dim_head), dtype=dtype)
            b_k = self.param('b_k', b_init, (num_heads, self.dim_head), dtype=dtype)
            b_v = self.param('b_v', b_init, (num_heads, self.dim_head), dtype=dtype)
            b_o = self.param('b_o', b_init, (self.dim,), dtype=dtype)

        # Linear projections for queries
        q = jnp.einsum('...li,ihd->...lhd', inputs_q, w_q)  # (..., seq_len_q, num_heads, dim_head)
        k = jnp.einsum('...li,ihd->...lhd', inputs_kv, w_k)  # (..., seq_len_kv, num_heads, dim_head)
        v = jnp.einsum('...li,ihd->...lhd', inputs_kv, w_v)  # (..., seq_len_kv, num_heads, dim_head)

        if self.use_bias:
            k += b_k
            v += b_v
            q += b_q 

        # Scaled dot-product attention
        scaling = 1 / jnp.sqrt(self.dim_head).astype(dtype)
        attn_scores = jnp.einsum('...qhd,...khd->...hqk', q * scaling, k * scaling * self.alpha_att)  # (..., num_heads, seq_len_q, seq_len_kv)
        # attn_scores = attn_scores * scaling
        attn_weights = nn.softmax(attn_scores, axis=-1)

        # Compute context vectors
        context = jnp.einsum('...hqk,...khd->...qhd', attn_weights, v)  # (..., seq_len_q, num_heads, dim_head)

        # Output projection
        out = jnp.einsum('...lhd,hdm->...lm', context, w_o)  # (..., seq_len_q, dim)
        if self.use_bias:
            out += b_o

        return out


class TransformerPayneModelWave(nn.Module):
    """Predicting on the single wave point."""
    dim: int = 256
    dim_ff_multiplier: int = 4
    no_tokens: int = 16
    no_layers: int = 16
    dim_head: int = 32
    out_dim: int = 1
    input_dim: int = 95
    min_period: float = 1e-6
    max_period: float = 10
    bias_dense: bool = False
    bias_attention: bool = False
    activation_fn: str = "gelu"
    output_activation_fn: str = "linear"
    init_att_q: str = "si"
    init_att_o: str = "si"
    emb_init: str = "si"
    ff_init: str = "si"
    head_init: str = "si"
    sigma: float = 1.0
    alpha_emb: float = 1.0
    alpha_att: float = 1.0
    reference_depth: int = None
    reference_width: int = None

    @nn.compact
    def __call__(self, x, train=False):
        atmospheric_parameters, wavelength = x

        activation_fn = _activation_functions_dict[self.activation_fn]
        output_activation_fn = _activation_functions_dict[self.output_activation_fn]

        num_heads = self.dim // self.dim_head
        residual_scaling = 1.0 if self.reference_depth is None else (self.no_layers / self.reference_depth)**(-0.5)

        enc_w = frequency_encoding(
            wavelength,
            min_period=self.min_period,
            max_period=self.max_period,
            dimension=self.dim
        )
        # Make sure that from now on we are working with float32
        enc_w = enc_w.astype(jnp.float32)
        atmospheric_parameters = atmospheric_parameters.astype(jnp.float32)

        enc_w = enc_w[None, ...]
        enc_p = ParametersEmbedding(
            dim=self.dim,
            input_dim=self.input_dim,
            no_tokens=self.no_tokens, 
            activation_fn=activation_fn, 
            use_bias=self.bias_dense,
            init_type=self.emb_init,
            sigma=self.sigma,
            alpha=self.alpha_emb
            )(atmospheric_parameters, train)

        # print(enc_p.shape, enc_w.shape)
        # [batch_sizes…, length, features]
        x = enc_w

        enc_p = nn.RMSNorm(name=f"norm_p", use_scale=False)(enc_p)
        for i in range(self.no_layers):
            # MHA
            _x = nn.RMSNorm(name=f"norm_1_L{i}", use_scale=False)(x)
            _x = MHA(dim=self.dim, 
                dim_head=self.dim_head, 
                use_bias=self.bias_attention,
                init_att_q=self.init_att_q,
                init_att_o=self.init_att_o,
                sigma=self.sigma,
                alpha_att=self.alpha_att
                )(inputs_q=_x, inputs_kv=enc_p)

            x = x + residual_scaling * _x
            # MLP
            _x = nn.RMSNorm(name=f"norm_2_L{i}", use_scale=False)(x)
            _x = FeedForward(dim=self.dim, 
                dim_ff_multiplier=self.dim_ff_multiplier, 
                activation_fn=activation_fn,
                init_type=self.ff_init,
                sigma=self.sigma
                )(_x, train)
            
            x = x + residual_scaling * _x
        
        x = nn.RMSNorm(name="decoder_norm", use_scale=False)(x)
        x = x[0]
        x = PredictionHead(dim=self.dim, 
            out_dim=self.out_dim, 
            use_bias=self.bias_dense,
            activation_fn=activation_fn, 
            output_activation_fn=output_activation_fn,
            init_type=self.head_init,
            sigma=self.sigma
            )(x, train)
        # ---
        x = x.at[1].set(10 ** x[1])
        x = x.at[0].set(x[0]*x[1])
        return x
    
class TransformerPayneModel(nn.Module):
    dim: int = 256
    dim_ff_multiplier: int = 4
    no_tokens: int = 16
    no_layers: int = 16
    dim_head: int = 32
    out_dim: int = 1
    input_dim: int = 95
    min_period: float = 1e-6
    max_period: float = 10
    bias_dense: bool = False
    bias_attention: bool = False
    activation_fn: str = "gelu"
    output_activation_fn: str = "linear"
    init_att_q: str = "si"
    init_att_o: str = "si"
    emb_init: str = "si"
    ff_init: str = "si"
    head_init: str = "si"
    sigma: float = 1.0
    alpha_emb: float = 1.0
    alpha_att: float = 1.0
    reference_depth: int = None
    reference_width: int = None

    @nn.compact
    def __call__(self, inputs, train):
        log_waves, p  = inputs

        TP = nn.vmap(
                    TransformerPayneModelWave, 
                    in_axes=((None, 0),),out_axes=0,
                    variable_axes={'params': None}, 
                    split_rngs={'params': False})
        
        x = TP(name="transformer_payne", 
                dim=self.dim, 
                dim_ff_multiplier=self.dim_ff_multiplier, 
                no_tokens=self.no_tokens, 
                no_layers=self.no_layers, 
                dim_head=self.dim_head, 
                out_dim=self.out_dim,
                input_dim=self.input_dim,
                min_period=self.min_period, 
                max_period=self.max_period,
                bias_dense=self.bias_dense,
                bias_attention=self.bias_attention,
                activation_fn=self.activation_fn, 
                output_activation_fn=self.output_activation_fn,
                init_att_q=self.init_att_q,
                init_att_o=self.init_att_o,
                emb_init=self.emb_init,
                ff_init=self.ff_init,
                head_init=self.head_init,
                sigma=self.sigma,
                alpha_emb=self.alpha_emb,
                alpha_att=self.alpha_att,
                reference_depth=self.reference_depth,
                reference_width=self.reference_width
                )((p, log_waves))
        return x


class TransformerPayne(SpectrumEmulator[ArrayLike]):

    def __init__(self, model_definition: ArchitectureDefinition):
        self.model_definition = model_definition
        # When restoring the state the first thing is to intialize a model
        if self.model_definition.architecture == ARCHITECTURE_NAME:
            # put architecture parameters in the model
            self.model = TransformerPayneModel(**self.model_definition.architecture_parameters)
        else:
            raise ValueError(f"Architecture {self.model_definition.architecture} not supported")
        
    @classmethod
    def download(cls, cache_path: str = ".cache"):
        from dataclasses import asdict
        from transformer_payne.huggingface_config import HUGGINGFACE_CONFIG
        
        
        hf_config = HUGGINGFACE_CONFIG.get(HF_CONFIG_NAME)
        repository_id, filename = hf_config.get(REPOSITORY_ID_KEY), hf_config.get(FILENAME_KEY)

        if os.path.isdir(cache_path):
            if os.path.isdir(os.path.join(cache_path, HF_CONFIG_NAME)):
                try:
                    model_architecture = ArchitectureDefinition.from_file(os.path.join(cache_path, HF_CONFIG_NAME, filename))
                    return cls(model_architecture)
                except ValueError as e:
                    warnings.warn(str(e) + " Downloading the model from HuggingFace.")
        else:
            warnings.warn("Path " + cache_path + " does not exist. Creating one")
            os.mkdir(cache_path)

        model_architecture = download_hf_model(repository_id, filename)
        serialization_path = os.path.join(cache_path, HF_CONFIG_NAME)
        if not os.path.isdir(serialization_path):
            os.mkdir(serialization_path)
        model_architecture.serialize(os.path.join(serialization_path, filename))
        return cls(model_architecture)

    @classmethod
    def from_file(cls, filepath: str):
        if os.path.isfile(filepath):
            try:
                model_architecture = ArchitectureDefinition.from_file(filepath)
                return cls(model_architecture)
            except ValueError as e:
                warnings.warn(str(e) + " Model is corrupted. Try downloading the model from HuggingFace using TransformerPayne.download()")
        else:
            warnings.warn("File " + filepath + " does not exist. Try downloading the model from HuggingFace using TransformerPayne.download()")

    @property
    def parameter_names(self) -> List[str]:
        """Get labels of spectrum model parameters

        Returns:
            List[str]:
        """
        return self.model_definition.spectral_parameters
    
    @property
    def stellar_parameter_names(self) -> List[str]:
        """Get names of stellar parameters (no parameters related to geometry, e.g. mu)

        Returns:
            List[str]:
        """
        return self.model_definition.spectral_parameters[:-1]
    
    @property
    def min_parameters(self) -> ArrayLike:
        """Minimum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return self.model_definition.min_spectral_parameters
    
    @property
    def min_stellar_parameters(self) -> ArrayLike:
        """Minimum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return self.model_definition.min_spectral_parameters[:-1]
    
    @property
    def max_parameters(self) -> ArrayLike:
        """Maximum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return self.model_definition.max_spectral_parameters
    
    @property
    def max_stellar_parameters(self) -> ArrayLike:
        """Maximum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return self.model_definition.max_spectral_parameters[:-1]

    @property
    def number_of_labels(self) -> int:
        """Number of labels for the spectrum model

        Returns:
            int:
        """
        return len(self.max_parameters)
    
    def is_in_bounds(self, parameters: ArrayLike) -> bool:
        """Check if parameters are within the bounds of the spectrum model

        Args:
            parameters (ArrayLike):

        Returns:
            bool:
        """
        return jnp.all(parameters >= self.min_parameters[:-1]) and jnp.all(parameters <= self.max_parameters[:-1])
    
    @property
    def solar_parameters(self) -> ArrayLike:
        """Solar parameters for the spectrum model

        Returns:
            ArrayLike:
        """
        return self.model_definition.solar_parameters[:-1]
    
    def to_parameters(self, parameter_values: Dict[str, Any] = None, relative: bool = True) -> ArrayLike:
        """Convert passed values to the accepted parameters format

        Args:
            parameter_values (Dict[str, Any], optional): parameter values in the format of {'parameter_name': value}. Unset parameters will be set to solar values.
            relative (bool, optional): if True, the values are treated as relative to the solar values for the abundaces 

        Returns:
            ArrayLike:
        """
        
        if not parameter_values:
            return self.solar_parameters
        
        # Initialize parameters with solar values
        parameters = np.array(self.solar_parameters)

        if parameter_values:
            # Convert parameter names to indices for direct access
            parameter_indices = {label: i for i, label in enumerate(self.stellar_parameter_names)}
            
            for label, value in parameter_values.items():
                # Get the index of the parameter
                idx = parameter_indices[label]
                
                # Adjust the value if relative is True and it's an abundance parameter
                if relative and self.model_definition.abundance_parameters[idx]:
                    parameters[idx] = value + self.solar_parameters[idx]
                else:
                    # Directly set the value if not relative or not an abundance parameter
                    parameters[idx] = value
        
        if not (jnp.all(parameters >= self.min_stellar_parameters) and jnp.all(parameters <= self.max_stellar_parameters)):
            warnings.warn("Possible exceeding parameter bonds - extrapolating.")
        
        return parameters

    def from_relative_parameters(self, relative_parameters: ArrayLike) -> ArrayLike:
        """Convert relative parameters to the accepted parameters format

        Args:
            relative_parameters (ArrayLike): relative parameter values

        Returns:
            ArrayLike: absolute parameter values
        """
        # just copy not abundances and increase abundances with solar values, vectorised numpy
        return jnp.where(self.model_definition.abundance_parameters[:-1], relative_parameters + self.solar_parameters, relative_parameters)

    def set_group_of_abundances_relative_to_solar(self, 
                                                    parameters: ArrayLike,
                                                    value: float,
                                                    group: List[str]) -> ArrayLike:
        """Update a group of abundance parameters relative to the solar values

        Args:
            parameters (ArrayLike): current parameters
            value (float): value to set for the group
            group (List[str]): list of parameter names to update

        Returns:
            ArrayLike: updated parameters
        """
        parameters = parameters.copy()
        # Convert parameter names to indices for direct access
        parameter_indices = {label: i for i, label in enumerate(self.stellar_parameter_names)}
        
        for label in group:
            # Get the index of the parameter
            idx = parameter_indices.get(label, None)
            if idx is not None:
                # Set the value if it's an abundance parameter
                if self.model_definition.abundance_parameters[idx]:
                    parameters[idx] = value + self.solar_parameters[idx]
            else:
                warnings.warn(f"Parameter {label} not found in the model definition")
        
        return parameters

    def set_abundances_relative_to_arbitrary_element(
        self, 
        parameters: ArrayLike, 
        value: float, 
        set_element: Union[str, List[str]], 
        reference_element: str = "Fe"
    ) -> ArrayLike:
        """Update a group of abundance parameters relative to the solar ratios

        Args:
            parameters (ArrayLike): current parameters
            value (float): value to set for the element(s)
            set_element (Union[str, List[str]]): element(s) to set the abundance for
            reference_element (str, optional): element to set the abundance relative to. Defaults to "Fe".

        Returns:
            ArrayLike: updated parameters
        """
        parameters = parameters.copy()
        # Ensure set_element is always a list for uniform processing
        if isinstance(set_element, str):
            set_element = [set_element]

        # Convert parameter names to indices for direct access
        parameter_indices = {label: i for i, label in enumerate(self.stellar_parameter_names)}

        # Get the index of the reference parameter
        idx_reference = parameter_indices.get(reference_element, None)
        if idx_reference is None or not self.model_definition.abundance_parameters[idx_reference]:
            warnings.warn(f"Reference parameter {reference_element} not an abundance parameter or not found in the model definition")
            return parameters

        # Iterate over each element in the set_element list
        for elem in set_element:
            idx_set = parameter_indices.get(elem, None)
            if idx_set is not None and self.model_definition.abundance_parameters[idx_set]:
                # Apply the abundance adjustment logic
                solar_relative = self.solar_parameters[idx_set] - self.solar_parameters[idx_reference]
                current_ref = parameters[idx_reference]
                parameters[idx_set] = current_ref + solar_relative + value
            else:
                warnings.warn(f"Parameter {elem} not an abundance parameter or not found in the model definition")

        return parameters

    def flux(self, log_wavelengths: ArrayLike, spectral_parameters: ArrayLike, mus_number: int = 10) -> ArrayLike:
        roots, weights = np.polynomial.legendre.leggauss(mus_number)
        roots = (roots + 1) / 2
        weights /= 2
        return _flux(roots, weights, self.intensity, log_wavelengths, spectral_parameters)


    # https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods
    # following an advice to create helper function that is to be jitted
    def intensity(self, log_wavelengths: ArrayLike, mu: float, spectral_parameters: ArrayLike) -> ArrayLike:
        """Calculate the intensity for given wavelengths and mus

        Args:
            log_wavelengths (ArrayLike): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight
            spectral_parameters (ArrayLike): an array of predefined stellar parameters

        Returns:
            ArrayLike: monochromatic intensities corresponding to passed wavelengths [erg/s/cm3/angstrom]
        """
        return _intensity(self, log_wavelengths, mu, spectral_parameters)

    def __call__(self, log_wavelengths: ArrayLike, mu: float, spectral_parameters: ArrayLike) -> ArrayLike:
        return self.intensity(log_wavelengths, mu, spectral_parameters)


@partial(jax.jit, static_argnums=(0, 1, 2))
def _flux(roots: ArrayLike, weights: ArrayLike, intensity_method: callable,
          log_wavelengths: ArrayLike, spectral_parameters: ArrayLike):
    roots, weights = jnp.array(roots), jnp.array(weights)
    return 2*jnp.pi*jnp.sum(
        jax.vmap(intensity_method, in_axes=(None, 0, None))(log_wavelengths, roots, spectral_parameters)*
        roots[:, jnp.newaxis, jnp.newaxis]*weights[:, jnp.newaxis, jnp.newaxis],
        axis=0
    )


@partial(jax.jit, static_argnums=(0,))
def _intensity(tp, log_wavelengths, mu, spectral_parameters):
    p_all = jnp.concatenate([spectral_parameters, jnp.atleast_1d(mu)], axis=0)
    p_all = (p_all-tp.min_parameters)/(tp.max_parameters-tp.min_parameters)
    return tp.model.apply({"params":freeze(tp.model_definition.emulator_weights)}, (log_wavelengths, p_all), train=False)