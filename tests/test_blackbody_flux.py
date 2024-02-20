from tpayne.blackbody_flux import BlackbodyFlux
import numpy as np
import pytest


@pytest.fixture
def blackbody_flux_instance():
    yield BlackbodyFlux()

class TestBlackbodyFlux:
    def test_is_in_bounds_under_lower(self, blackbody_flux_instance):
        assert blackbody_flux_instance.is_in_bounds(-1.0) == False

    def test_is_in_bounds(self, blackbody_flux_instance):
        assert blackbody_flux_instance.is_in_bounds(10000) == True
