from abc import abstractmethod
from tpayne.spectrum_emulator import SpectrumEmulator
from typing import TypeVar


T = TypeVar("T")

class IntensityEmulator(SpectrumEmulator[T]):
    @abstractmethod
    def intensity(self, log_wavelengths: T, mu: float, parameters: T) -> T:
        raise NotImplementedError
