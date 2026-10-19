"""Shared helpers for loading OpenQDC datasets (TopoBench and MACE pipelines).

OpenQDC stores every dataset in the same flat layout (``atomic_inputs``,
``position_idx_range``, ``energies`` with one column per QM method), but the
unit label of a dataset is not always right (e.g. QM9 is declared as hartree
while its numbers are kcal/mol). :class:`OpenQDCSource` therefore loads data in
its *declared* units (no conversion inside OpenQDC) and converts on read from
the *stored* units, which are taken from a verified registry or detected by
comparing total energies with the sum of free-atom energies.
"""

import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]

HARTREE_PER_UNIT = {
    "hartree": 1.0,
    "ev": 1.0 / 27.211386246,
    "kcal/mol": 1.0 / 627.5094740631,
    "kj/mol": 1.0 / 2625.4996394799,
}
ANGSTROM_PER_UNIT = {"ang": 1.0, "bohr": 0.529177210903, "nm": 10.0}

# Units every pipeline works in.
ENERGY_UNIT = "ev"
DISTANCE_UNIT = "ang"
FORCE_UNIT = "ev/ang"

# Approximate non-relativistic free-atom total energies (Hartree). Only used to
# tell whether a column holds total energies and in which unit.
ATOM_REFERENCE_HARTREE = {
    1: -0.50,
    2: -2.90,
    3: -7.48,
    4: -14.67,
    5: -24.65,
    6: -37.84,
    7: -54.59,
    8: -75.07,
    9: -99.73,
    10: -128.93,
    11: -162.29,
    12: -200.10,
    13: -242.35,
    14: -289.36,
    15: -341.26,
    16: -398.11,
    17: -460.15,
    35: -2572.5,
}

# Stored energy units checked by hand (they win over auto-detection).
KNOWN_STORED_ENERGY_UNITS = {"QM9": "kcal/mol"}
# Datasets whose energy columns are not total energies.
NOT_TOTAL_ENERGY = {
    "QM8": "columns are excitation energies (E1, E2 x CC2/PBE0/PBE0.1/CAM)"
}
# Datasets that cannot be used at all.
UNUSABLE = {"QM7": "remote OpenQDC copy is corrupted (all-zero data)"}
# Used when a dataset is not downloaded so its elements are unknown.
FALLBACK_ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]


def default_cache_dir() -> str:
    """Return the OpenQDC cache directory used by the pipelines.

    Returns
    -------
    str
        ``$OPENQDC_CACHE_DIR`` if set, else ``<repo>/datasets/openqdc``.
    """
    env = os.environ.get("OPENQDC_CACHE_DIR")
    if env:
        return str(Path(env).expanduser().resolve())
    return str(PROJECT_ROOT / "datasets" / "openqdc")


def available_datasets() -> dict[str, type]:
    """Map canonical OpenQDC potential-energy dataset names to their classes.

    Returns
    -------
    dict[str, type]
        Class name to OpenQDC dataset class.
    """
    import openqdc

    return dict(openqdc.AVAILABLE_POTENTIAL_DATASETS)


def canonical_name(name: str) -> str:
    """Return the OpenQDC class name for a case-insensitive dataset name.

    Parameters
    ----------
    name : str
        Dataset name, e.g. ``"qm9"``.

    Returns
    -------
    str
        Canonical class name, e.g. ``"QM9"``.

    Raises
    ------
    ValueError
        If the dataset is unknown.
    """
    datasets = available_datasets()
    by_lower = {k.lower(): k for k in datasets}
    if name.lower() not in by_lower:
        raise ValueError(
            f"Unknown OpenQDC dataset '{name}'. Available: {sorted(datasets)}"
        )
    return by_lower[name.lower()]


def resolve_energy_method(names: list[str], spec) -> int:
    """Turn an energy-method index or name into an index.

    Parameters
    ----------
    names : list[str]
        Names of the energy columns of the dataset.
    spec : int or str
        Column index (``3`` or ``"3"``) or a column name (case-insensitive).

    Returns
    -------
    int
        Column index.

    Raises
    ------
    ValueError
        If the index is out of range or the name does not match one column.
    """
    if isinstance(spec, int) or str(spec).lstrip("-").isdigit():
        idx = int(spec)
        if not 0 <= idx < len(names):
            raise ValueError(
                f"energy_method must be in [0, {len(names) - 1}], got {idx}."
            )
        return idx
    hits = [i for i, n in enumerate(names) if n.lower() == str(spec).lower()]
    if len(hits) != 1:
        raise ValueError(
            f"energy method '{spec}' matches {len(hits)} columns of {names}."
        )
    return hits[0]


def _prepare_class(name: str, cache_dir: str):
    import openqdc
    from openqdc.utils.io import set_cache_dir

    set_cache_dir(str(cache_dir))
    return getattr(openqdc, canonical_name(name))


def is_cached(name: str, cache_dir: str) -> bool:
    """Check whether a dataset is already downloaded in ``cache_dir``.

    Parameters
    ----------
    name : str
        Dataset name.
    cache_dir : str
        OpenQDC cache directory.

    Returns
    -------
    bool
        True if all arrays are present locally.
    """
    cls = _prepare_class(name, cache_dir)
    try:
        return bool(cls.no_init().is_cached())
    except Exception:
        return False


class OpenQDCSource:
    """An OpenQDC dataset read in declared units and converted on access.

    Parameters
    ----------
    name : str
        OpenQDC dataset name (case-insensitive), e.g. ``"QM9"``.
    cache_dir : str
        OpenQDC cache directory; the dataset is downloaded there if missing.
    energy_unit : str
        Unit of the energies returned by :meth:`get`.
    distance_unit : str
        Unit of the positions returned by :meth:`get`.
    stored_energy_unit : str, optional
        True unit of the stored energies. ``None`` uses the verified registry
        and otherwise detects it (see :meth:`detect_stored_energy_unit`).
    stored_distance_unit : str, optional
        True unit of the stored positions; ``None`` uses the declared unit,
        and ``"auto"`` detects it from nearest-neighbour distances.
    energy_method : int
        Column used for unit detection.

    Attributes
    ----------
    energy_kind : str
        ``"total"``, ``"not_total"`` or ``"unknown"``.
    """

    def __init__(
        self,
        name: str,
        cache_dir: str,
        energy_unit: str = ENERGY_UNIT,
        distance_unit: str = DISTANCE_UNIT,
        stored_energy_unit: str | None = None,
        stored_distance_unit: str | None = None,
        energy_method: int = 0,
    ) -> None:
        self.name = canonical_name(name)
        cls = _prepare_class(self.name, cache_dir)
        self.declared_energy_unit = str(
            getattr(cls, "__energy_unit__", "hartree")
        ).lower()
        self.declared_distance_unit = str(
            getattr(cls, "__distance_unit__", "ang")
        ).lower()
        # Loading in declared units makes OpenQDC skip every conversion.
        self.ds = cls(
            energy_unit=self.declared_energy_unit,
            distance_unit=self.declared_distance_unit,
            array_format="numpy",
            cache_dir=str(cache_dir),
            skip_statistics=True,
        )
        n_atoms = self.ds.data["position_idx_range"]
        if len(n_atoms) == 0 or n_atoms[:, 1].max() == 0:
            raise RuntimeError(
                f"OpenQDC dataset '{self.name}' has no atom positions "
                "(corrupted download)."
            )

        self.energy_unit = energy_unit.lower()
        self.distance_unit = distance_unit.lower()
        self.energy_kind = "unknown"
        if stored_energy_unit is not None:
            self.stored_energy_unit = stored_energy_unit.lower()
            self.energy_kind = "total"
        elif self.name in KNOWN_STORED_ENERGY_UNITS:
            self.stored_energy_unit = KNOWN_STORED_ENERGY_UNITS[self.name]
            self.energy_kind = "total"
        else:
            detected, _ = self.detect_stored_energy_unit(energy_method)
            self.stored_energy_unit = detected or self.declared_energy_unit
            if detected is not None:
                self.energy_kind = "total"
        if self.name in NOT_TOTAL_ENERGY:
            self.energy_kind = "not_total"

        if stored_distance_unit == "auto":
            detected_d, _ = self.detect_stored_distance_unit()
            stored_distance_unit = detected_d
        self.stored_distance_unit = (
            stored_distance_unit or self.declared_distance_unit
        ).lower()

        self.energy_scale = (
            HARTREE_PER_UNIT[self.stored_energy_unit]
            / HARTREE_PER_UNIT[self.energy_unit]
        )
        self.distance_scale = (
            ANGSTROM_PER_UNIT[self.stored_distance_unit]
            / ANGSTROM_PER_UNIT[self.distance_unit]
        )

    def __len__(self) -> int:
        return len(self.ds)

    @property
    def n_methods(self) -> int:
        """Number of energy columns."""
        return int(self.ds.data["energies"].shape[1])

    @property
    def method_names(self) -> list[str]:
        """Names of the energy columns (OpenQDC labels may be empty)."""
        names = getattr(self.ds, "energy_target_names", None)
        if names is not None and len(names) == self.n_methods:
            return [str(n) for n in names]
        return [str(m) for m in self.ds.energy_methods]

    def get(
        self, i: int, target: str = "formation_energies", method: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Read one molecule.

        Parameters
        ----------
        i : int
            Molecule index.
        target : str
            ``"formation_energies"``, ``"energies"`` or
            ``"per_atom_formation_energies"``.
        method : int
            Energy column.

        Returns
        -------
        tuple
            ``(atomic_numbers, charges, positions, energy)`` with positions in
            ``distance_unit`` and the energy in ``energy_unit``.
        """
        entry = self.ds[int(i)]
        return (
            np.asarray(entry.atomic_numbers, dtype=int),
            np.asarray(entry.charges, dtype=float),
            np.asarray(entry.positions, dtype=float) * self.distance_scale,
            float(entry[target][method]) * self.energy_scale,
        )

    def get_forces(self, i: int, force_method: int = 0) -> np.ndarray:
        """Read the forces of one molecule in ``energy_unit/distance_unit``.

        Forces share the stored energy and distance units of the dataset, so
        they are converted with ``energy_scale / distance_scale`` (eV/Angstrom
        with the default units).

        Parameters
        ----------
        i : int
            Molecule index.
        force_method : int
            Force column.

        Returns
        -------
        np.ndarray
            Forces, shape ``[n_atoms, 3]``.

        Raises
        ------
        ValueError
            If the dataset has no forces.
        """
        if "forces" not in self.ds.data:
            raise ValueError(f"{self.name} has no forces.")
        entry = self.ds[int(i)]
        forces = np.asarray(entry.forces, dtype=float)[..., force_method]
        return forces * self.energy_scale / self.distance_scale

    def elements(self) -> list[int]:
        """Atomic numbers present in the dataset.

        Returns
        -------
        list[int]
            Sorted atomic numbers.
        """
        z = self.ds.data["atomic_inputs"]
        found: set[int] = set()
        for lo in range(0, len(z), 5_000_000):
            found |= {int(a) for a in np.unique(z[lo : lo + 5_000_000, 0])}
        return sorted(found)

    def _sample(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(len(self), min(n, len(self)), replace=False))

    def detect_stored_energy_unit(
        self, method: int = 0, n: int = 2000
    ) -> tuple[str | None, float | None]:
        """Detect the true unit of a column of total energies.

        The median ratio between the stored energy and the sum of free-atom
        energies is ~1 when the right unit is applied.

        Parameters
        ----------
        method : int
            Energy column.
        n : int
            Number of molecules sampled.

        Returns
        -------
        tuple
            ``(unit, ratio)``: ``unit`` is None if no unit makes the column
            look like total energies; ``ratio`` is the median ratio when the
            declared unit is assumed (None if not enough molecules).
        """
        data = self.ds.data
        ratios = []
        for i in self._sample(n):
            lo, hi = data["position_idx_range"][i]
            z = data["atomic_inputs"][lo:hi, 0].astype(int)
            energy = float(data["energies"][i, method])
            if (
                not np.isfinite(energy)
                or energy == 0.0
                or any(a not in ATOM_REFERENCE_HARTREE for a in z)
            ):
                continue
            reference = sum(ATOM_REFERENCE_HARTREE[a] for a in z)
            ratios.append(
                energy
                * HARTREE_PER_UNIT[self.declared_energy_unit]
                / reference
            )
        if len(ratios) < 20:
            return None, None
        ratio = float(np.median(ratios))
        for unit, factor in HARTREE_PER_UNIT.items():
            implied = (
                ratio * factor / HARTREE_PER_UNIT[self.declared_energy_unit]
            )
            if 0.9 < implied < 1.1:
                return unit, ratio
        return None, ratio

    def median_neighbour_distance(self, n: int = 100) -> float | None:
        """Median nearest-neighbour distance in the stored distance unit.

        Parameters
        ----------
        n : int
            Number of molecules sampled.

        Returns
        -------
        float or None
            The median, or None if no suitable molecule was found.
        """
        data = self.ds.data
        values = []
        for i in self._sample(n * 3):
            lo, hi = data["position_idx_range"][i]
            if not 2 <= hi - lo <= 500:
                continue
            pos = np.asarray(data["atomic_inputs"][lo:hi, -3:], dtype=float)
            dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            np.fill_diagonal(dist, np.inf)
            values.append(np.median(dist.min(axis=1)))
            if len(values) >= n:
                break
        return float(np.median(values)) if values else None

    def detect_stored_distance_unit(self) -> tuple[str | None, float | None]:
        """Detect the true distance unit from nearest-neighbour distances.

        Returns
        -------
        tuple
            ``(unit, median_distance_in_declared_unit)``; ``unit`` is None if
            no unit gives a plausible (0.85-1.4 Angstrom) median.
        """
        median = self.median_neighbour_distance()
        if median is None:
            return None, None
        # The declared unit is tried first so a correct label is never changed.
        units = [self.declared_distance_unit, *ANGSTROM_PER_UNIT]
        for unit in units:
            if 0.85 < median * ANGSTROM_PER_UNIT[unit] < 1.4:
                return unit, median
        return None, median


def inspect_dataset(
    name: str, cache_dir: str, download: bool = False, energy_method: int = 0
) -> dict:
    """Collect metadata used to write a dataset config.

    Class-level metadata (declared units, number of methods, forces) needs no
    data. Elements, size and unit checks need the dataset to be downloaded.

    Parameters
    ----------
    name : str
        Dataset name.
    cache_dir : str
        OpenQDC cache directory.
    download : bool
        Download the dataset if it is not cached.
    energy_method : int
        Energy column used for the unit checks.

    Returns
    -------
    dict
        Metadata, JSON serialisable.
    """
    name = canonical_name(name)
    cls = _prepare_class(name, cache_dir)
    info = {
        "name": name,
        "declared_energy_unit": str(getattr(cls, "__energy_unit__", "")),
        "declared_distance_unit": str(getattr(cls, "__distance_unit__", "")),
        "n_methods": len(getattr(cls, "__energy_methods__", [])),
        "has_forces": len(getattr(cls, "__force_mask__", [])) > 0,
        "unusable": UNUSABLE.get(name),
        "not_total": NOT_TOTAL_ENERGY.get(name),
        "cached": False,
    }
    if info["unusable"]:
        return info
    if not (is_cached(name, cache_dir) or download):
        return info
    src = OpenQDCSource(name, cache_dir, energy_method=energy_method)
    _, ratio = (
        (None, None)
        if name in KNOWN_STORED_ENERGY_UNITS
        else src.detect_stored_energy_unit(energy_method)
    )
    dist_unit, nn_dist = src.detect_stored_distance_unit()
    info.update(
        cached=True,
        n_molecules=len(src),
        n_methods=src.n_methods,
        method_names=src.method_names,
        elements=src.elements(),
        max_atoms=int(src.ds.data["n_atoms"].max()),
        stored_energy_unit=src.stored_energy_unit,
        energy_kind=src.energy_kind,
        energy_ratio_vs_declared=ratio,
        stored_distance_unit=dist_unit or src.declared_distance_unit,
        distance_check_ok=dist_unit is not None,
        median_neighbour_distance=nn_dist,
    )
    return info
