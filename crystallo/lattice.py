"""Unit cells, metric tensors and interplanar spacings for all seven crystal systems."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Lattice"]


@dataclass(frozen=True)
class Lattice:
    """A unit cell. Lengths in angstroms, angles in degrees."""

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    # ---- constructors -------------------------------------------------------
    @classmethod
    def cubic(cls, a: float) -> "Lattice":
        return cls(a, a, a)

    @classmethod
    def tetragonal(cls, a: float, c: float) -> "Lattice":
        return cls(a, a, c)

    @classmethod
    def orthorhombic(cls, a: float, b: float, c: float) -> "Lattice":
        return cls(a, b, c)

    @classmethod
    def hexagonal(cls, a: float, c: float) -> "Lattice":
        return cls(a, a, c, 90.0, 90.0, 120.0)

    @classmethod
    def rhombohedral(cls, a: float, alpha: float) -> "Lattice":
        return cls(a, a, a, alpha, alpha, alpha)

    @classmethod
    def monoclinic(cls, a: float, b: float, c: float, beta: float) -> "Lattice":
        return cls(a, b, c, 90.0, beta, 90.0)

    # ---- geometry -----------------------------------------------------------
    @property
    def metric(self) -> np.ndarray:
        """Real-space metric tensor G_ij = a_i . a_j."""
        al, be, ga = (math.radians(x) for x in (self.alpha, self.beta, self.gamma))
        a, b, c = self.a, self.b, self.c
        return np.array([
            [a * a, a * b * math.cos(ga), a * c * math.cos(be)],
            [a * b * math.cos(ga), b * b, b * c * math.cos(al)],
            [a * c * math.cos(be), b * c * math.cos(al), c * c],
        ])

    @property
    def volume(self) -> float:
        """Cell volume in cubic angstroms."""
        det = np.linalg.det(self.metric)
        # relative tolerance: a flat (coplanar) cell has det ~ 0 up to round-off
        if det <= 1e-10 * (self.a * self.b * self.c) ** 2:
            raise ValueError("cell angles do not describe a valid 3D cell")
        return math.sqrt(det)

    @property
    def reciprocal_metric(self) -> np.ndarray:
        return np.linalg.inv(self.metric)

    def matrix(self) -> np.ndarray:
        """Rows are the lattice vectors in a Cartesian frame (a along x, b in xy)."""
        al, be, ga = (math.radians(x) for x in (self.alpha, self.beta, self.gamma))
        a_vec = [self.a, 0.0, 0.0]
        b_vec = [self.b * math.cos(ga), self.b * math.sin(ga), 0.0]
        cx = self.c * math.cos(be)
        cy = self.c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
        cz = math.sqrt(max(self.c**2 - cx**2 - cy**2, 0.0))
        return np.array([a_vec, b_vec, [cx, cy, cz]])

    def d_spacing(self, h: int, k: int, l: int) -> float:
        """Interplanar spacing d_hkl in angstroms."""
        hkl = np.array([h, k, l], dtype=float)
        if not hkl.any():
            raise ValueError("(000) has no spacing")
        return 1.0 / math.sqrt(hkl @ self.reciprocal_metric @ hkl)

    def bragg_two_theta(self, h: int, k: int, l: int, wavelength: float = 1.5406) -> float | None:
        """Diffraction angle 2-theta in degrees, or ``None`` if the reflection is beyond reach."""
        s = wavelength / (2.0 * self.d_spacing(h, k, l))
        if s > 1.0:
            return None
        return 2.0 * math.degrees(math.asin(s))

    def angle_between_planes(self, hkl1, hkl2) -> float:
        """Angle in degrees between the normals of two lattice planes."""
        g = self.reciprocal_metric
        v1, v2 = np.asarray(hkl1, float), np.asarray(hkl2, float)
        cosang = (v1 @ g @ v2) / math.sqrt((v1 @ g @ v1) * (v2 @ g @ v2))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
