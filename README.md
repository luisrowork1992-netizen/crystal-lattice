# crystal-lattice

![tests](https://github.com/luisrowork1992-netizen/crystal-lattice/actions/workflows/tests.yml/badge.svg)

`crystallo` is a small crystallography toolkit for **unit-cell geometry, densities, packing fractions and powder X-ray diffraction peak positions**.

## Features

- `Lattice` for all seven crystal systems: metric tensor, cell volume, Cartesian lattice vectors
- d-spacings, Bragg angles, and angles between planes for any (hkl), triclinic included
- Prototype structures: simple cubic, BCC, FCC, diamond, rocksalt, CsCl, and HCP
- Density from the cell contents, and hard-sphere packing fraction
- Structure factors with systematic absences, and powder patterns with multiplicities and Lorentz-polarisation-weighted relative intensities

## Install

```bash
git clone https://github.com/luisrowork1992-netizen/crystal-lattice.git
cd crystal-lattice
pip install -e .
```

## Quick start

```python
import crystallo as cr

si = cr.diamond("Si", 5.431)
print(si.density)                       # 2.329 g/cm3
print(si.packing_fraction())            # 0.340
print(si.is_allowed(2, 0, 0))           # False  (diamond glide absence)

for p in si.powder_pattern()[:3]:       # Cu K-alpha1 by default
    print(p.hkl, round(p.two_theta, 2)) # (1,1,1) 28.44, (2,2,0) 47.3, (3,1,1) 56.12

lat = cr.Lattice(5.0, 6.0, 7.0, 80, 95, 105)   # triclinic
print(lat.volume, lat.d_spacing(1, 2, 3))
```

## Notebook

[`notebooks/xrd_patterns.ipynb`](notebooks/xrd_patterns.ipynb) covers four topics:

- a density and packing table for eight materials
- simulated XRD patterns showing the FCC, BCC and diamond selection rules
- the effect of a two-atom basis, comparing NaCl with CsCl
- how the Cu (111) peak shifts under strain

## Tests

```bash
pip install pytest
pytest
```

The tests check results against reference values:

- densities of Cu (8.93), Fe (7.87), Si (2.329) and NaCl (2.16 g/cm³)
- analytic packing fractions: π/6, π√3/8, π/(3√2), π√3/16
- the triclinic volume formula
- hexagonal d-spacings
- Si peaks at 28.44°, 47.30° and 56.12° 2θ
- the strongest Cu peak, (111) at 43.3°
- selection rules for FCC, BCC, diamond, rocksalt and CsCl

## Limitations

- Atomic form factors are taken as constant (*f* = Z), and there are no Debye–Waller or absorption corrections.
- Intensities are qualitative. Peak positions are exact.

## References

- B. D. Cullity and S. R. Stock, *Elements of X-Ray Diffraction*, 3rd ed., Pearson.
- C. Giacovazzo (ed.), *Fundamentals of Crystallography*, 3rd ed., Oxford University Press.

## License

MIT
