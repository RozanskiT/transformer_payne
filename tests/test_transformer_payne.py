from transformer_payne.architecture_definition import ArchitectureDefinition
from transformer_payne.transformer_payne import TransformerPayne
import numpy as np
import pytest
import joblib
import os

UNDER_LOWER_BOUND = np.array([ 2.60205999,  1.        , -1.        , -1.3       , -3.8       ,
       -3.8       , -3.8       , -4.8       , -3.8       , -4.8       ,
       -3.8       , -4.8       , -3.8       , -4.8       , -3.8       ,
       -4.8       , -3.8       , -4.8       , -3.8       , -4.8       ,
       -3.8       , -4.8       , -3.8       , -4.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -3.8       ,
       -3.8       , -3.8       , -3.8       , -3.8       , -1.        ],
                             dtype=np.float32)

OVER_UPPER_BOUND = np.array([4.90308999, 6.        , 6.        , 1.3       , 2.3       ,
       2.3       , 2.3       , 3.3       , 2.3       , 3.3       ,
       2.3       , 3.3       , 2.3       , 3.3       , 2.3       ,
       3.3       , 2.3       , 3.3       , 2.3       , 3.3       ,
       2.3       , 3.3       , 2.3       , 3.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.3       ,
       2.3       , 2.3       , 2.3       , 2.3       , 2.        ],
                            dtype=np.float32)

ONE_UNDER_LOWER_BOUND = np.array([ 3.40205999,  2.        ,  0.        , -0.3       , -2.8       ,
       -2.8       , -2.8       , -3.8       , -2.8       , -3.8       ,
       -2.8       , -3.8       , -2.8       , -3.8       , -2.8       ,
       -3.8       , -2.8       , -3.8       , -2.8       , -3.8       ,
       -2.8       , -3.8       , -2.8       , -3.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       , -2.8       ,
       -2.8       , -2.8       , -2.8       , -2.8       ,  0.        ], dtype=np.float32)

ONE_OVER_UPPER_BOUND = np.array([3.90308999, 5.        , 5.        , 0.3       , 1.3       ,
       1.3       , 1.3       , 2.3       , 1.3       , 2.3       ,
       1.3       , 2.3       , 1.3       , 2.3       , 1.3       ,
       2.3       , 1.3       , 2.3       , 1.3       , 2.3       ,
       1.3       , 2.3       , 1.3       , 2.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 3.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.3       ,
       1.3       , 1.3       , 1.3       , 1.3       , 1.        ], dtype=np.float32)

IN_BOUNDS = np.array([ 3.75257499,  3.5       ,  2.5       ,  0.        , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      , -0.75      ,
       -0.75      , -0.75      , -0.75      , -0.75      ,  0.5       ], dtype=np.float32)


@pytest.fixture
def transformer_payne_instance(datadir):
    model_architecture = ArchitectureDefinition.from_dict_config(joblib.load(os.path.join(datadir, 'transformer_payne.pkl')))
    yield TransformerPayne(model_architecture)

class TestTransformerPayne:
    def test_is_in_bounds_all_under_lower(self, transformer_payne_instance):
        assert transformer_payne_instance.is_in_bounds(UNDER_LOWER_BOUND) == False

    def test_is_in_bounds_all_over_upper(self, transformer_payne_instance):
        assert transformer_payne_instance.is_in_bounds(OVER_UPPER_BOUND) == False

    def test_is_in_bounds_one_under_lower(self, transformer_payne_instance):
        assert transformer_payne_instance.is_in_bounds(ONE_UNDER_LOWER_BOUND) == False

    def test_is_in_bounds_one_over_upper(self, transformer_payne_instance):
        assert transformer_payne_instance.is_in_bounds(ONE_OVER_UPPER_BOUND) == False

    def test_is_in_bounds(self, transformer_payne_instance):
        assert transformer_payne_instance.is_in_bounds(IN_BOUNDS) == True

    @pytest.mark.skip(reason="To be changed")
    def test_to_parameters_default_all_default(self, transformer_payne_instance):
        assert np.all(np.isclose(transformer_payne_instance.to_parameters(),
                                 np.array([3.7617023675414125, 4.44, 1., 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                                 1e-5)
                      )

    @pytest.mark.skip(reason="To be changed")
    def test_to_parameters_default_abundances_default(self, transformer_payne_instance):
        assert np.all(np.isclose(transformer_payne_instance.to_parameters(logteff=3.65, logg=4.5, mu=1.0),
                                 np.array([3.65, 4.5, 1., 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                                 1e-5)
                      )

    @pytest.mark.skip(reason="To be changed")
    def test_raise_exception_when_not_in_bounds(self, transformer_payne_instance):
        with pytest.raises(ValueError):
            transformer_payne_instance.to_parameters(logteff=3.2, logg=-4.5, mu=1.0)

    @pytest.mark.skip(reason="To be changed")
    def test_to_parameters_abundance_array(self, transformer_payne_instance):
        assert np.all(np.isclose(
            transformer_payne_instance.to_parameters(logteff=3.65, logg=4.5, mu=1.0,
                                 abundances=np.array(
                                     [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                                 )),
            np.array([3.65, 4.5, 1.,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                     dtype=np.float32),
        ))

    @pytest.mark.skip(reason="To be changed")
    def test_to_parameters_some_abundances_as_dict(self, transformer_payne_instance):
        assert np.all(np.isclose(transformer_payne_instance.to_parameters(
            logteff=3.65, logg=4.5, mu=1.0,
            abundances={"Be": 0.4, "Pa": 0.3}),
            np.array([3.65, 4.5, 1., 0.0, 0.4, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.3, 0.0], dtype=np.float32),
            1e-5)
        )