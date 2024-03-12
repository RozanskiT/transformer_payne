from typing import Any, Dict, List, Union
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
    
    @partial(jax.jit, static_argnums=(0, 3))
    def flux(self, log_wavelengths: ArrayLike, spectral_parameters: ArrayLike, mus_number: int = 10) -> ArrayLike:
        roots, weights = np.polynomial.legendre.leggauss(mus_number)
        roots = (roots + 1) / 2
        weights /= 2
        
        roots, weights = jnp.array(roots), jnp.array(weights)
        return 2*jnp.pi*jnp.sum(
            jax.vmap(self.intensity, in_axes=(None, 0, None))(log_wavelengths, roots, spectral_parameters)*
            roots[:, jnp.newaxis, jnp.newaxis]*weights[:, jnp.newaxis, jnp.newaxis],
            axis=0
        )

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

# correct version

@partial(jax.jit, static_argnums=(0,))
def _intensity(tp, log_wavelengths, mu, spectral_parameters):
    p_all = jnp.concatenate([spectral_parameters, jnp.atleast_1d(mu)], axis=0)
    p_all = (p_all-tp.min_parameters)/(tp.max_parameters-tp.min_parameters)
    return tp.model.apply({"params":freeze(tp.model_definition.emulator_weights)}, (log_wavelengths, p_all), train=False)