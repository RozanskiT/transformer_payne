from typing import Any, Dict, List
from dataclasses import asdict, dataclass, fields

@dataclass
class ArchitectureDefinition:
    emulator_weights: str
    architecture: str
    architecture_parameters: Dict[str, Any]
    spectral_parameters: List[str]
    min_spectral_parameters: List[float]
    max_spectral_parameters: List[float]
    solar_parameters: List[float]
    abundance_parameters: List[float]
    tag: str

    @classmethod
    def from_dict_config(cls, model_dict: Dict[str, Any]):
        for field in fields(cls):
            if field.name not in model_dict:
                raise ValueError("Passed dictionary has no key" + field.name)
            
        return cls(**model_dict)
    
    @classmethod
    def from_file(cls, filename: str):
        import joblib
        import os

        if not os.path.isfile(filename):
            raise ValueError("File " + filename + " does not exist!")
        try:
            model_dict = joblib.load(filename)
            return cls.from_dict_config(model_dict)
        except IOError:
            raise ValueError("File " + filename + " is corrupted!")

    def serialize(self, filename: str):
        import joblib
        joblib.dump(asdict(self), filename)
