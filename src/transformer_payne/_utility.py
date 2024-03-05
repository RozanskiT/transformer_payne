from typing import Union
import numpy as np

def vac_to_air(lambda_vac: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert vacuum wavelength to air wavelength using the IAU standard formula by Donald Morton (2000),
    which is based on the work of Birch and Downs (1994).

    Reference:
    Morton, D. C. (2000). Atomic Data for Resonance Absorption Lines. I. Wavelengths Longward of the Lyman Limit.
    The Astrophysical Journal Supplement Series, 130: 403.
    Birch, K. P., & Downs, M. J. (1994). Correction to the Updated Edlén Equation for the Refractive Index of Air.
    Metrologia, 31: 315.

    The formula applies to dry air at 1 atm pressure and 15 degree Celsius with 0.045% CO2 by volume.

    Parameters:
    lambda_vac (float | np.ndarray): The vacuum wavelength(s) in Angstroms.

    Returns:
    (float | np.ndarray): The air wavelength(s) in Angstroms.
    """
    # Calculate s based on the vacuum wavelength
    s = 10**4 / lambda_vac
    
    # Calculate the refractive index, n
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    
    # Convert vacuum wavelength to air wavelength
    lambda_air = lambda_vac / n
    
    return lambda_air

def air_to_vac(lambda_air: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert air wavelength to vacuum wavelength using the inverse transformation derived by N. Piskunov,
    which is intended to be a reverse of the Morton (2000) and Birch and Downs (1994) formulas.

    Reference:
    Piskunov, N. (no publication date available, check VALD webpage). Conversion formula for air wavelength to vacuum wavelength.
    Birch, K. P., & Downs, M. J. (1994). Correction to the Updated Edlén Equation for the Refractive Index of Air.
    Metrologia, 31: 315.

    The formula is intended for use with dry air at 1 atm pressure and 15 degree Celsius with 0.045% CO2 by volume.

    Parameters:
    lambda_air (float | np.ndarray): The air wavelength(s) in Angstroms.

    Returns:
    (float | np.ndarray): The vacuum wavelength(s) in Angstroms.
    """
    # Calculate s based on the air wavelength
    s = 10**4 / lambda_air
    
    # Calculate the refractive index, n, using the Piskunov's formula
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    
    # Convert air wavelength to vacuum wavelength
    lambda_vac = lambda_air * n
    
    return lambda_vac

METALS = ["Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
        "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
        "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
        "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
        "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]

ATOMIC_SYMBOLS = ["H", "He"] + METALS