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
       -3.8       , -3.8       , -3.8       , -3.8       ],
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
       2.3       , 2.3       , 2.3       , 2.3       ],
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
       -2.8       , -2.8       , -2.8       , -2.8       ], dtype=np.float32)

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
       1.3       , 1.3       , 1.3       , 1.3       ], dtype=np.float32)

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
       -0.75      , -0.75      , -0.75      , -0.75      ], dtype=np.float32)


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
        relative = transformer_payne_instance.from_relative_parameters(IN_BOUNDS)
        assert transformer_payne_instance.is_in_bounds(relative) == True

    def test_to_parameters_default_all_default(self, transformer_payne_instance):
        assert np.all(np.isclose(transformer_payne_instance.to_parameters(),
                                 np.array([ 3.76170237,  4.44      ,  1.        , 10.93      ,  1.05      ,
        1.38      ,  2.7       ,  8.39      ,  7.78      ,  8.66      ,
        4.56      ,  7.84      ,  6.17      ,  7.53      ,  6.37      ,
        7.51      ,  5.36      ,  7.14      ,  5.5       ,  6.18      ,
        5.08      ,  6.31      ,  3.17      ,  4.9       ,  4.        ,
        5.64      ,  5.39      ,  7.45      ,  4.92      ,  6.23      ,
        4.21      ,  4.6       ,  2.88      ,  3.58      ,  2.29      ,
        3.33      ,  2.56      ,  3.25      ,  2.6       ,  2.92      ,
        2.21      ,  2.58      ,  1.42      ,  1.92      , -5.        ,
        1.84      ,  1.12      ,  1.66      ,  0.94      ,  1.77      ,
        1.6       ,  2.        ,  1.        ,  2.19      ,  1.51      ,
        2.24      ,  1.07      ,  2.17      ,  1.13      ,  1.7       ,
        0.58      ,  1.45      , -5.        ,  1.        ,  0.52      ,
        1.11      ,  0.28      ,  1.14      ,  0.51      ,  0.93      ,
        0.        ,  1.08      ,  0.06      ,  0.88      , -0.17      ,
        1.11      ,  0.23      ,  1.25      ,  1.38      ,  1.64      ,
        1.01      ,  1.13      ,  0.9       ,  2.        ,  0.65      ,
       -5.        , -5.        , -5.        , -5.        , -5.        ,
       -5.        ,  0.06      , -5.        , -0.52], dtype=np.float32),
                                 1e-5)
                      )

    def test_to_parameters_default_abundances_default(self, transformer_payne_instance):
        assert np.all(np.isclose(transformer_payne_instance.to_parameters(dict(logteff=3.65, logg=4.5)),
                                 np.array([ 3.65,  4.5      ,  1.        , 10.93      ,  1.05      ,
        1.38      ,  2.7       ,  8.39      ,  7.78      ,  8.66      ,
        4.56      ,  7.84      ,  6.17      ,  7.53      ,  6.37      ,
        7.51      ,  5.36      ,  7.14      ,  5.5       ,  6.18      ,
        5.08      ,  6.31      ,  3.17      ,  4.9       ,  4.        ,
        5.64      ,  5.39      ,  7.45      ,  4.92      ,  6.23      ,
        4.21      ,  4.6       ,  2.88      ,  3.58      ,  2.29      ,
        3.33      ,  2.56      ,  3.25      ,  2.6       ,  2.92      ,
        2.21      ,  2.58      ,  1.42      ,  1.92      , -5.        ,
        1.84      ,  1.12      ,  1.66      ,  0.94      ,  1.77      ,
        1.6       ,  2.        ,  1.        ,  2.19      ,  1.51      ,
        2.24      ,  1.07      ,  2.17      ,  1.13      ,  1.7       ,
        0.58      ,  1.45      , -5.        ,  1.        ,  0.52      ,
        1.11      ,  0.28      ,  1.14      ,  0.51      ,  0.93      ,
        0.        ,  1.08      ,  0.06      ,  0.88      , -0.17      ,
        1.11      ,  0.23      ,  1.25      ,  1.38      ,  1.64      ,
        1.01      ,  1.13      ,  0.9       ,  2.        ,  0.65      ,
       -5.        , -5.        , -5.        , -5.        , -5.        ,
       -5.        ,  0.06      , -5.        , -0.52], dtype=np.float32),
                                 1e-5)
                      )
