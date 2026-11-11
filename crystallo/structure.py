"""Crystal structures, densities, packing fractions and powder-diffraction peak lists.

Scattering uses a constant form factor (f = Z) per atom, which is enough to get
systematic absences and rough relative intensities right. It is not meant
for quantitative Rietveld work.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .lattice import Lattice

__all__ = [
    "Site",
    "Crystal",
    "Peak",
    "simple_cubic",
    "bcc",
    "fcc",
    "diamond",
    "rocksalt",
    "cesium_chloride",
    "hcp",
    "AVOGADRO",
    "ATOMIC",
]

AVOGADRO = 6.02214076e23

# (atomic number, standard atomic weight) for the elements used in examples/tests
ATOMIC = {
    "H": (1, 1.008), "C": (6, 12.011), "N": (7, 14.007), "O": (8, 15.999),
    "Na": (11, 22.990), "Mg": (12, 24.305), "Al": (13, 26.982), "Si": (14, 28.085),
    "Cl": (17, 35.45), "K": (19, 39.098), "Ti": (22, 47.867), "Cr": (24, 51.996),
    "Fe": (26, 55.845), "Ni": (28, 58.693), "Cu": (29, 63.546), "Zn": (30, 65.38),
    "Ge": (32, 72.630), "Mo": (42, 95.95), "Ag": (47, 107.87), "Cs": (55, 132.91),
    "W": (74, 183.84), "Pt": (78, 195.08), "Au": (79, 196.97), "Pb": (82, 207.2),
}


@dataclass(frozen=True)
class Site:
    element: str
    frac: tuple[float, float, float]


@dataclass(frozen=True)
class Peak:
    hkl: tuple[int, int, int]
    d: float
    two_theta: float
    multiplicity: int
    intensity: float  # normalised to 100 for the strongest peak


@dataclass
class Crystal:
    name: str
    lattice: Lattice
    sites: list[Site] = field(default_factory=list)
    atomic_radius: float | None = None  # for packing fraction of single-element cells

    @property
    def atoms_per_cell(self) -> int:
        return len(self.sites)

    @property
    def cell_mass_g(self) -> float:
        return sum(ATOMIC[s.element][1] for s in self.sites) / AVOGADRO

    @property
    def density(self) -> float:
        """Mass density in g/cm^3."""
        return self.cell_mass_g / (self.lattice.volume * 1e-24)

    def packing_fraction(self) -> float:
        """Fraction of the cell volume filled by touching hard spheres."""
        if self.atomic_radius is None:
            raise ValueError("atomic_radius not defined for this structure")
        vol = self.atoms_per_cell * 4.0 / 3.0 * math.pi * self.atomic_radius**3
        return vol / self.lattice.volume

    def structure_factor(self, h: int, k: int, l: int) -> complex:
        F = 0j
        for s in self.sites:
            x, y, z = s.frac
            F += ATOMIC[s.element][0] * np.exp(2j * math.pi * (h * x + k * y + l * z))
        return complex(F)

    def is_allowed(self, h: int, k: int, l: int, tol: float = 1e-6) -> bool:
        return abs(self.structure_factor(h, k, l)) > tol

    def powder_pattern(self, wavelength: float = 1.5406, two_theta_max: float = 120.0, hkl_max: int = 6) -> list[Peak]:
        """Allowed reflections, merged by d-spacing, with a Lorentz-polarisation intensity."""
        groups: dict[float, list[tuple[int, int, int]]] = {}
        for hkl in itertools.product(range(-hkl_max, hkl_max + 1), repeat=3):
            if hkl == (0, 0, 0) or not self.is_allowed(*hkl):
                continue
            tt = self.lattice.bragg_two_theta(*hkl, wavelength=wavelength)
            if tt is None or tt > two_theta_max:
                continue
            key = round(self.lattice.d_spacing(*hkl), 5)
            groups.setdefault(key, []).append(hkl)

        peaks = []
        for d, members in groups.items():
            rep = max(members)  # largest-index-first representative, e.g. (1,1,1)
            tt = self.lattice.bragg_two_theta(*rep, wavelength=wavelength)
            th = math.radians(tt / 2)
            lp = (1 + math.cos(2 * th) ** 2) / (math.sin(th) ** 2 * math.cos(th))
            I = sum(abs(self.structure_factor(*m)) ** 2 for m in members) * lp
            peaks.append(Peak(_canonical(rep), d, tt, len(members), I))
        peaks.sort(key=lambda p: p.two_theta)
        if peaks:
            top = max(p.intensity for p in peaks)
            peaks = [Peak(p.hkl, p.d, p.two_theta, p.multiplicity, 100 * p.intensity / top) for p in peaks]
        return peaks


def _canonical(hkl):
    return tuple(sorted((abs(i) for i in hkl), reverse=True))


# ---- prototype structures ------------------------------------------------
_FCC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]


def simple_cubic(element: str, a: float) -> Crystal:
    return Crystal(f"{element} (sc)", Lattice.cubic(a), [Site(element, (0, 0, 0))], atomic_radius=a / 2)


def bcc(element: str, a: float) -> Crystal:
    sites = [Site(element, (0, 0, 0)), Site(element, (0.5, 0.5, 0.5))]
    return Crystal(f"{element} (bcc)", Lattice.cubic(a), sites, atomic_radius=a * math.sqrt(3) / 4)


def fcc(element: str, a: float) -> Crystal:
    sites = [Site(element, p) for p in _FCC]
    return Crystal(f"{element} (fcc)", Lattice.cubic(a), sites, atomic_radius=a * math.sqrt(2) / 4)


def diamond(element: str, a: float) -> Crystal:
    sites = [Site(element, p) for p in _FCC]
    sites += [Site(element, (x + 0.25, y + 0.25, z + 0.25)) for x, y, z in _FCC]
    return Crystal(f"{element} (diamond)", Lattice.cubic(a), sites, atomic_radius=a * math.sqrt(3) / 8)


def rocksalt(cation: str, anion: str, a: float) -> Crystal:
    sites = [Site(cation, p) for p in _FCC]
    sites += [Site(anion, ((x + 0.5) % 1, y, z)) for x, y, z in _FCC]
    return Crystal(f"{cation}{anion} (rocksalt)", Lattice.cubic(a), sites)


def cesium_chloride(cation: str, anion: str, a: float) -> Crystal:
    sites = [Site(cation, (0, 0, 0)), Site(anion, (0.5, 0.5, 0.5))]
    return Crystal(f"{cation}{anion} (CsCl)", Lattice.cubic(a), sites)


def hcp(element: str, a: float, c: float | None = None) -> Crystal:
    """Hexagonal close-packed; ideal c/a = sqrt(8/3) if ``c`` is omitted."""
    c = a * math.sqrt(8 / 3) if c is None else c
    sites = [Site(element, (1 / 3, 2 / 3, 0.25)), Site(element, (2 / 3, 1 / 3, 0.75))]
    return Crystal(f"{element} (hcp)", Lattice.hexagonal(a, c), sites, atomic_radius=a / 2)
