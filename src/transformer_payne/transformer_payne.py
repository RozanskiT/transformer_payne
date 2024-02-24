from typing import Any, Dict, List
from transformer_payne.architecture_definition import ArchitectureDefinition
from transformer_payne.download import download_hf_model
from transformer_payne.exceptions import JAXWarning
from transformer_payne.intensity_emulator import IntensityEmulator
from transformer_payne.configuration import DEFAULT_CACHE_PATH, REPOSITORY_ID_KEY, FILENAME_KEY
from functools import partial
import warnings
import os

HF_CONFIG_NAME = "TPAYNE"
ARCHITECTURE_NAME = "TransformerPayneIntensities"

try:
    import jax.numpy as jnp
    from jax.typing import ArrayLike
    import jax
    from flax import linen as nn
    from flax.core.frozen_dict import freeze
    
except ImportError:
    import numpy as jnp
    from numpy.typing import ArrayLike
    warnings.warn("Please install JAX and Flax to use TransformerPayne.", JAXWarning)

def frequency_encoding(x, min_period, max_period, dimension):
    periods = jnp.logspace(jnp.log10(min_period), jnp.log10(max_period), num=dimension)
    
    y = jnp.sin(2*jnp.pi/periods*x)
    return y

class TransformerPayneModelWave(nn.Module):
    no_layer: int = 16
    no_token: int = 16
    d_att: int = 256
    d_ff: int = 1024
    no_head: int = 8
    out_dimensionality: int = 2

    @nn.compact
    def __call__(self, x):
        p, w = x
        enc_w = frequency_encoding(w, min_period=1e-6, max_period=10, dimension=self.d_att)
        enc_w = enc_w[None, ...]
        p = nn.gelu(nn.Dense(4*self.d_att,name="encoder_0")(p))
        p = nn.Dense(self.no_token*self.d_att, name="encoder_1")(p)
        enc_p = jnp.reshape(p, (self.no_token, self.d_att))
        
        # print(enc_p.shape, enc_w.shape)
        # [batch_sizes…, length, features]
        x_pre = enc_w
        x_post = enc_w
        for i in range(self.no_layer):
            # MHA
            _x = x_post + nn.MultiHeadDotProductAttention(num_heads=self.no_head,name=f"cond_mha_{i}")(inputs_q=x_post,
                                                                inputs_kv=nn.LayerNorm(name=f"norm_0_L{i}")(enc_p))
            x_pre = x_pre + _x
            x_post = nn.LayerNorm(name=f"norm_1_L{i}")(_x)
            # MLP
            _x = x_post + nn.Dense(self.d_att, name=f"cond_dense_1_L{i}")(nn.gelu(nn.Dense(self.d_ff, name=f"cond_dense_0_L{i}")(x_post)))
            
            x_pre = x_pre + _x
            x_post = nn.LayerNorm(name=f"norm_2_L{i}")(_x)
        
        x_pre = nn.LayerNorm(name="decoder_norm")(x_pre)
        x = x_pre + x_post
        x = nn.gelu(nn.Dense(256, name="decoder_0")(x[0]))
        x = nn.Dense(self.out_dimensionality, name="decoder_1")(x)
        # ---
        x = x.at[1].set(10 ** x[1])
        x = x.at[0].set(x[0]*x[1])
        return x
    
class TransformerPayneModel(nn.Module):
    no_layer: int = 16
    no_token: int = 16
    d_att: int = 256
    d_ff: int = 1024
    no_head: int = 8
    out_dimensionality: int = 2

    @nn.compact
    def __call__(self, inputs, train):
        log_waves, p  = inputs
        DecManyWave = nn.vmap(
                    TransformerPayneModelWave, 
                    in_axes=((None, 0),),out_axes=0,
                    variable_axes={'params': None}, 
                    split_rngs={'params': False})
        
        x = DecManyWave(name="attention_emulator")((p, log_waves))
        return x


class TransformerPayne(IntensityEmulator[ArrayLike]):

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


    @property
    def parameter_names(self) -> List[str]:
        """Get labels of spectrum model parameters

        Returns:
            List[str]:
        """
        return self.model_definition.spectral_parameters
    
    @property
    def min_parameters(self) -> ArrayLike:
        """Minimum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return self.model_definition.min_spectral_parameters
    
    @property
    def max_parameters(self) -> ArrayLike:
        """Maximum values supported by the spectrum model

        Returns:
            ArrayLike:
        """
        return self.model_definition.max_spectral_parameters

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
        return jnp.all(parameters >= self.min_parameters) and jnp.all(parameters <= self.max_parameters)
    
    @property
    def solar_parameters(self) -> ArrayLike:
        """Solar parameters for the spectrum model

        Returns:
            ArrayLike:
        """
        return self.model_definition.solar_parameters
    
    def to_parameters(self, parameter_values: Dict[str, Any] = None) -> ArrayLike:
        """Convert passed values to the accepted parameters format

        Args:
            parameter_values (Dict[str, Any], optional): parameter values in the format of {'parameter_name': value}. Unset parameters will be set to solar values.

        Returns:
            ArrayLike:
        """
        
        if not parameter_values:
            return self.solar_parameters
        
        parameters = jnp.array([parameter_values.get(label, self.solar_parameters[i]) for i, label in enumerate(self.parameter_names)])
        
        if not self.is_in_bounds(parameters):
            warnings.warn("Possible exceeding parameter bonds - extrapolating.")
        
        return parameters

    # https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods
    # following an advice to create helper function that is to be jitted
    def intensity(self, log_wavelengths: ArrayLike, mu: float, spectral_parameters: ArrayLike) -> ArrayLike:
        """Calculate the intensity for given wavelengths and mus

        Args:
            log_wavelengths (ArrayLike): [log(angstrom)]
            mu (float): cosine of the angle between the star's radius and the line of sight
            spectral_parameters (ArrayLike): an array of predefined stellar parameters

        Returns:
            ArrayLike: intensities corresponding to passed wavelengths [erg/cm2/s/angstrom]
        """
        return _intensity(self, log_wavelengths, mu, spectral_parameters)

    def __call__(self, log_wavelengths: ArrayLike, mu: float, spectral_parameters: ArrayLike) -> ArrayLike:
        return self.intensity(log_wavelengths, mu, spectral_parameters)

# correct version

@partial(jax.jit, static_argnums=(0,))
def _intensity(tp, log_wavelengths, mu, spectral_parameters):
    p_all = jnp.empty(tp.number_of_labels, dtype=jnp.float32)
    p_all = p_all.at[-1].set(mu)
    p_all = p_all.at[:-1].set(spectral_parameters)
    return tp.model.apply({"params":freeze(tp.model_definition.emulator_weights)}, (log_wavelengths, p_all), train=False)