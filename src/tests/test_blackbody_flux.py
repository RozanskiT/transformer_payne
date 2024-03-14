from transformer_payne.blackbody import Blackbody
import numpy as np
import pytest


@pytest.fixture
def blackbody_flux_instance():
    yield Blackbody()

class TestBlackbodyFlux:
    def test_is_in_bounds_under_lower(self, blackbody_flux_instance):
        assert blackbody_flux_instance.is_in_bounds(-1.0) == False

    def test_is_in_bounds(self, blackbody_flux_instance):
        assert blackbody_flux_instance.is_in_bounds(10000) == True

    def test_to_parameters(self, blackbody_flux_instance):
        assert np.all(
            np.isclose(
                blackbody_flux_instance.to_parameters(dict(teff=10000)), np.array([10000])
                )
            )
        
    def test_to_parameters_out_of_bounds(self, blackbody_flux_instance):
        with pytest.raises(ValueError):
            blackbody_flux_instance.to_parameters(dict(teff=-1.0))
