from abc import abstractmethod
from tpayne.spectrum_emulator import SpectrumEmulator
from typing import TypeVar


T = TypeVar("T")

class FluxEmulator(SpectrumEmulator[T]):
    @abstractmethod
    def flux(self, log_wavelengths: T, parameters: T) -> T:
        raise NotImplementedError