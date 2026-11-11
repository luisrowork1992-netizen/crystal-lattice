import math

import pytest

import crystallo as cr


def test_cubic_volume_and_d_spacing():
    lat = cr.Lattice.cubic(4.0)
    assert lat.volume == pytest.approx(64.0)
    assert lat.d_spacing(1, 1, 1) == pytest.approx(4.0 / math.sqrt(3))
    assert lat.d_spacing(2, 0, 0) == pytest.approx(2.0)


def test_hexagonal_volume_and_spacing():
    a, c = 3.21, 5.21
    lat = cr.Lattice.hexagonal(a, c)
    assert lat.volume == pytest.approx(math.sqrt(3) / 2 * a * a * c)
    # 1/d^2 = 4/3 (h^2 + hk + k^2)/a^2 + l^2/c^2
    h, k, l = 1, 0, 1
    expected = 1 / math.sqrt(4 / 3 * (h * h + h * k + k * k) / a**2 + l**2 / c**2)
    assert lat.d_spacing(h, k, l) == pytest.approx(expected)


def test_triclinic_volume_formula():
    a, b, c, al, be, ga = 5.0, 6.0, 7.0, 80.0, 95.0, 105.0
    lat = cr.Lattice(a, b, c, al, be, ga)
    ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
    expected = a * b * c * math.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    assert lat.volume == pytest.approx(expected)
    # the Cartesian matrix must reproduce the same volume
    import numpy as np
    assert abs(np.linalg.det(lat.matrix())) == pytest.approx(expected)


def test_invalid_cell_raises():
    with pytest.raises(ValueError):
        cr.Lattice(1, 1, 1, 120, 120, 120).volume   # coplanar vectors
    with pytest.raises(ValueError):
        cr.Lattice(1, 1, 1, 130, 130, 130).volume   # impossible angles


def test_interplanar_angles_cubic():
    lat = cr.Lattice.cubic(3.0)
    assert lat.angle_between_planes((1, 0, 0), (1, 1, 0)) == pytest.approx(45.0)
    assert lat.angle_between_planes((1, 0, 0), (1, 1, 1)) == pytest.approx(54.7356, abs=1e-4)


def test_copper_density():
    assert cr.fcc("Cu", 3.615).density == pytest.approx(8.93, abs=0.02)


def test_silicon_and_iron_density():
    assert cr.diamond("Si", 5.431).density == pytest.approx(2.329, abs=0.005)
    assert cr.bcc("Fe", 2.8665).density == pytest.approx(7.87, abs=0.02)


def test_rocksalt_nacl_density():
    assert cr.rocksalt("Na", "Cl", 5.640).density == pytest.approx(2.16, abs=0.01)


@pytest.mark.parametrize(
    "builder, expected",
    [
        (lambda: cr.simple_cubic("Cu", 3.0), math.pi / 6),
        (lambda: cr.bcc("Fe", 3.0), math.pi * math.sqrt(3) / 8),
        (lambda: cr.fcc("Cu", 3.0), math.pi / (3 * math.sqrt(2))),
        (lambda: cr.diamond("Si", 5.0), math.pi * math.sqrt(3) / 16),
        (lambda: cr.hcp("Mg", 3.2), math.pi / (3 * math.sqrt(2))),
    ],
)
def test_packing_fractions(builder, expected):
    assert builder().packing_fraction() == pytest.approx(expected)


def test_fcc_selection_rule():
    cu = cr.fcc("Cu", 3.615)
    assert cu.is_allowed(1, 1, 1) and cu.is_allowed(2, 0, 0) and cu.is_allowed(2, 2, 0)
    assert not cu.is_allowed(1, 0, 0) and not cu.is_allowed(1, 1, 0) and not cu.is_allowed(2, 1, 0)


def test_bcc_selection_rule():
    fe = cr.bcc("Fe", 2.8665)
    assert fe.is_allowed(1, 1, 0) and fe.is_allowed(2, 0, 0) and fe.is_allowed(2, 1, 1)
    assert not fe.is_allowed(1, 0, 0) and not fe.is_allowed(1, 1, 1)


def test_diamond_forbids_200_and_222():
    si = cr.diamond("Si", 5.431)
    assert si.is_allowed(1, 1, 1) and si.is_allowed(2, 2, 0) and si.is_allowed(3, 1, 1)
    assert not si.is_allowed(2, 0, 0) and not si.is_allowed(2, 2, 2)


def test_silicon_powder_pattern_cu_kalpha():
    peaks = cr.diamond("Si", 5.431).powder_pattern()
    first = [(p.hkl, round(p.two_theta, 2)) for p in peaks[:3]]
    assert [h for h, _ in first] == [(1, 1, 1), (2, 2, 0), (3, 1, 1)]
    assert first[0][1] == pytest.approx(28.44, abs=0.02)
    assert first[1][1] == pytest.approx(47.30, abs=0.02)
    assert first[2][1] == pytest.approx(56.12, abs=0.02)
    assert peaks[0].multiplicity == 8


def test_copper_strongest_peak_is_111():
    peaks = cr.fcc("Cu", 3.615).powder_pattern()
    assert [p.hkl for p in peaks[:4]] == [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)]
    strongest = max(peaks, key=lambda p: p.intensity)
    assert strongest.hkl == (1, 1, 1)
    assert strongest.two_theta == pytest.approx(43.30, abs=0.05)


def test_nacl_111_weak_but_present():
    peaks = {p.hkl: p for p in cr.rocksalt("Na", "Cl", 5.640).powder_pattern()}
    assert (1, 1, 1) in peaks and (2, 0, 0) in peaks
    assert peaks[(1, 1, 1)].intensity < peaks[(2, 0, 0)].intensity


def test_cscl_has_100_reflection():
    cscl = cr.cesium_chloride("Cs", "Cl", 4.12)
    assert cscl.is_allowed(1, 0, 0)  # would be absent in bcc with identical atoms
