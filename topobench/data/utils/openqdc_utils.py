"""Shared helpers for loading OpenQDC datasets (TopoBench and MACE pipelines).

Every dataset is loaded through OpenQDC's own unit conversion
(``energy_unit="ev"``, ``distance_unit="ang"``). QM9 and QM7b source-unit
metadata is corrected before conversion. Forces are never read.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.data import covalent_radii
from filelock import FileLock, Timeout

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Units requested from OpenQDC for every dataset.
ENERGY_UNIT = "ev"
DISTANCE_UNIT = "ang"
# The QMX archives store kcal/mol; OpenQDC 0.1.2 declares Hartree instead.
SOURCE_ENERGY_UNITS = {"QM9": "kcal/mol", "QM7b": "kcal/mol"}

log = logging.getLogger(__name__)

# Datasets whose energy columns are not total energies.
NOT_TOTAL_ENERGY = {
    "QM8": "columns are excitation energies (E1, E2 x CC2/PBE0/PBE0.1/CAM)"
}
# Known caveats of the targets, beyond their units.
ENERGY_NOTES = {
    "MultixcQM9": "the energy columns are atomization-type energies (about "
    "-4 to -7 eV per atom), not total energies",
}
_NOTED: set[str] = set()
# Datasets that cannot be used at all.
UNUSABLE = {"QM7": "remote OpenQDC copy is corrupted (all-zero data)"}
# Used when a dataset is not downloaded so its elements are unknown.
FALLBACK_ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]


@contextmanager
def locked(path: str, what: str) -> Iterator[None]:
    """Hold a file lock, saying so when another job holds it.

    Parameters
    ----------
    path : str
        Lock file.
    what : str
        What the lock protects, used in the waiting message.

    Yields
    ------
    None
        Control while the lock is held.
    """
    lock = FileLock(str(path))
    try:
        lock.acquire(timeout=3)
    except Timeout:
        print(
            f"[wait] {what}: another job is busy with it, waiting ...",
            flush=True,
        )
        lock.acquire()
    try:
        yield
    finally:
        lock.release()


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


def method_names_for(name: str) -> list[str]:
    """Return the names of the energy columns of a dataset without loading data.

    Parameters
    ----------
    name : str
        Dataset name.

    Returns
    -------
    list[str]
        One name per energy column (may be empty strings if OpenQDC has none).
    """
    cls = available_datasets()[canonical_name(name)]
    methods = list(getattr(cls, "__energy_methods__", []))
    names = getattr(cls, "energy_target_names", None)
    if names is not None and len(names) == len(methods):
        return [str(n) for n in names]
    return [str(m) for m in methods]


def molecular_edges(
    positions: Any,
    numbers: Any,
    method: str = "bonds",
    bond_scale: float = 1.3,
    cutoff: float = 5.0,
) -> torch.Tensor:
    """Build the undirected edge index of one molecule from its geometry.

    This is the edge construction of ``OpenQDCDataset`` (and of the molecule
    viewer), so both always show the same graph.

    Parameters
    ----------
    positions : array-like
        Atom positions in Angstrom, shape ``[N, 3]``.
    numbers : array-like
        Atomic numbers, shape ``[N]``.
    method : str
        ``"bonds"``: two atoms are connected if their distance is below
        ``bond_scale`` times the sum of their covalent radii. ``"radius"``:
        connected if the distance is below ``cutoff`` (Angstrom).
    bond_scale : float
        Factor on the sum of covalent radii (``method="bonds"``).
    cutoff : float
        Distance cutoff in Angstrom (``method="radius"``).

    Returns
    -------
    torch.Tensor
        Edge index, shape ``[2, E]``, both directions included, no self-loops.
    """
    if method not in ("bonds", "radius"):
        raise ValueError("method must be 'bonds' or 'radius'.")
    pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32)
    z = torch.as_tensor(np.asarray(numbers), dtype=torch.long)
    dist = torch.cdist(pos, pos)
    if method == "bonds":
        r = torch.as_tensor(covalent_radii, dtype=pos.dtype)[z]
        adj = dist < bond_scale * (r[:, None] + r[None, :])
    else:
        adj = dist < cutoff
    adj.fill_diagonal_(False)
    return adj.nonzero().t().contiguous()


def split_molecules(
    ids: Any,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split molecules into train, validation and test sets by OpenQDC index.

    This is the one split used by both pipelines (TopoBench and MACE). The
    molecules are identified by their OpenQDC index and sorted first, so the
    split does not depend on the order in which a pipeline lists them: the same
    molecules, fractions and seed always give the same split.

    Parameters
    ----------
    ids : array-like
        OpenQDC indices of the molecules to split.
    valid_fraction : float
        Fraction of the molecules in the validation set.
    test_fraction : float
        Fraction of the molecules in the test set.
    seed : int
        Seed of the random permutation.

    Returns
    -------
    dict[str, numpy.ndarray]
        The OpenQDC indices of each split (keys ``train``, ``valid``, ``test``).
    """
    ids = np.sort(np.asarray(ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_valid = int(valid_fraction * len(ids))
    n_test = int(test_fraction * len(ids))
    return {
        "train": ids[perm[n_valid + n_test :]],
        "valid": ids[perm[:n_valid]],
        "test": ids[perm[n_valid : n_valid + n_test]],
    }


def split_positions(
    ids: Any,
    valid_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Give the split of :func:`split_molecules` as positions in a list.

    Parameters
    ----------
    ids : array-like
        OpenQDC indices of the molecules, in the order of the list.
    valid_fraction : float
        Fraction of the molecules in the validation set.
    test_fraction : float
        Fraction of the molecules in the test set.
    seed : int
        Seed of the random permutation.

    Returns
    -------
    dict[str, numpy.ndarray]
        For each split (``train``, ``valid``, ``test``) the positions in
        ``ids`` of its molecules.
    """
    ids = np.asarray(ids)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    split = split_molecules(ids, valid_fraction, test_fraction, seed)
    return {
        name: order[np.searchsorted(sorted_ids, members)]
        for name, members in split.items()
    }


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
                f"energy_method must be in [0, {len(names) - 1}], got {idx}. "
                f"First columns: {names[:8]}"
            )
        return idx
    hits = [i for i, n in enumerate(names) if n.lower() == str(spec).lower()]
    if len(hits) != 1:
        raise ValueError(
            f"energy method '{spec}' matches {len(hits)} columns of {names}."
        )
    return hits[0]


def energy_cache_suffix(name: str) -> str:
    """Separate corrected QM9/QM7b graphs and XYZ files from older exports.

    Parameters
    ----------
    name : str
        Dataset name (case-insensitive).

    Returns
    -------
    str
        Unit revision suffix for QM9/QM7b, otherwise an empty string.
    """
    return (
        "_kcalmol_ev_v1" if canonical_name(name) in SOURCE_ENERGY_UNITS else ""
    )


def _prepare_class(name: str, cache_dir: str):
    """Point OpenQDC at ``cache_dir`` and return the dataset class.

    Parameters
    ----------
    name : str
        Dataset name (case-insensitive).
    cache_dir : str
        OpenQDC cache directory.

    Returns
    -------
    type
        The OpenQDC dataset class.
    """
    import openqdc
    from openqdc.utils.io import set_cache_dir

    set_cache_dir(str(cache_dir))
    name = canonical_name(name)
    cls = getattr(openqdc, name)
    source_unit = SOURCE_ENERGY_UNITS.get(name)
    if source_unit is not None and cls.__energy_unit__ != source_unit:
        if version("openqdc") != "0.1.2" or cls.__energy_unit__ != "hartree":
            raise RuntimeError(
                f"Verify {name} source energy units for OpenQDC "
                f"{version('openqdc')} before loading: declared "
                f"{cls.__energy_unit__!r}, expected {source_unit!r}."
            )
        # Scope the metadata correction to this reader, not OpenQDC globally.
        cls = type(cls.__name__, (cls,), {"__energy_unit__": source_unit})
    return cls


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


def files_are_consistent(
    files: dict[str, Path],
    extra: list[Path],
    shapes: dict[str, tuple],
    types: dict[str, str],
) -> bool:
    """Check that the files of a downloaded dataset are complete and agree.

    A download killed midway leaves a truncated file. The check compares sizes
    and row counts: the number of atoms must match the last atom range, the
    number of energies (and forces) the number of molecules, and the props file
    must end with the pickle stop byte.

    Parameters
    ----------
    files : dict[str, Path]
        Array name (``atomic_inputs``, ``position_idx_range``, ``energies`` and
        optionally ``forces``) to file.
    extra : list[Path]
        Extra files; the first is the pickle with the molecule properties.
    shapes : dict[str, tuple]
        OpenQDC array shapes (first entry -1).
    types : dict[str, str]
        OpenQDC array dtypes.

    Returns
    -------
    bool
        True if all files exist, are complete and consistent.
    """
    if not all(f.exists() for f in [*files.values(), *extra]):
        return False
    rows = {}
    for key, path in files.items():
        unit = int(np.prod(shapes[key][1:])) * np.dtype(types[key]).itemsize
        if unit == 0 or path.stat().st_size % unit:
            return False
        rows[key] = path.stat().st_size // unit
    ranges = np.memmap(files["position_idx_range"], dtype=np.int32, mode="r")
    ranges = ranges.reshape(-1, 2)
    if len(ranges) == 0 or rows["atomic_inputs"] != int(ranges[:, 1].max()):
        return False
    if rows["energies"] != len(ranges):
        return False
    if "forces" in rows and rows["forces"] != rows["atomic_inputs"]:
        return False
    with open(extra[0], "rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"."


def _remove_broken_download(dataset_class: type) -> None:
    """Remove the files of an interrupted download so OpenQDC fetches them again.

    OpenQDC writes files in place and trusts any file that exists, so a download
    killed midway leaves a truncated file that later fails to load (or loads
    wrong data). Nothing is removed when no file exists or all are consistent.

    Parameters
    ----------
    dataset_class : type
        OpenQDC dataset class (the cache directory is already set).
    """
    inst = dataset_class.no_init()
    folder = Path(inst.preprocess_path)
    wrapper = inst.dataset_wrapper
    files = {k: folder / wrapper.add_extension(k) for k in inst.data_keys}
    extra = [folder / f for f in wrapper._extra_files]
    existing = [f for f in [*files.values(), *extra] if f.exists()]
    if not existing:
        return
    if files_are_consistent(files, extra, inst.data_shapes, inst.data_types):
        return
    print(
        f"[repair] {dataset_class.__name__}: incomplete files found (interrupted "
        f"download?); removing them to download again: {folder}",
        flush=True,
    )
    for f in existing:
        f.unlink()


class OpenQDCSource:
    """An OpenQDC dataset loaded with OpenQDC's own eV / Angstrom conversion.

    Parameters
    ----------
    name : str
        OpenQDC dataset name (case-insensitive), e.g. ``"QM9"``.
    cache_dir : str
        OpenQDC cache directory; the dataset is downloaded there if missing.

    Attributes
    ----------
    not_total_reason : str or None
        Why the energies are not total energies (None if nothing is known).
    """

    def __init__(self, name: str, cache_dir: str) -> None:
        self.name = canonical_name(name)
        cls = _prepare_class(self.name, cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        # One process at a time loads (and, if needed, downloads) a dataset:
        # jobs started together wait here and then find the files.
        lock_file = Path(cache_dir) / f".{self.name}.lock"
        with locked(lock_file, f"{self.name} data (download/loading)"):
            _remove_broken_download(cls)
            self.ds = cls(
                energy_unit=ENERGY_UNIT,
                distance_unit=DISTANCE_UNIT,
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
        self.not_total_reason = NOT_TOTAL_ENERGY.get(self.name)
        if self.name in ENERGY_NOTES and self.name not in _NOTED:
            _NOTED.add(self.name)
            log.warning("%s energies: %s.", self.name, ENERGY_NOTES[self.name])

    def __len__(self) -> int:
        """Return the number of molecules.

        Returns
        -------
        int
            Number of molecules in the dataset.
        """
        return len(self.ds)

    @property
    def n_methods(self) -> int:
        """Return the number of energy columns.

        Returns
        -------
        int
            Number of energy columns per molecule.
        """
        return int(self.ds.data["energies"].shape[1])

    @property
    def method_names(self) -> list[str]:
        """Return the names of the energy columns.

        OpenQDC labels may be empty strings.

        Returns
        -------
        list[str]
            One name per energy column.
        """
        names = getattr(self.ds, "energy_target_names", None)
        if names is not None and len(names) == self.n_methods:
            return [str(n) for n in names]
        return [str(m) for m in self.ds.energy_methods]

    def get(
        self, i: int, target: str = "energies", method: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Read one molecule.

        Parameters
        ----------
        i : int
            Molecule index.
        target : str
            ``"energies"`` (total energy), ``"formation_energies"`` or
            ``"per_atom_formation_energies"``.
        method : int
            Energy column.

        Returns
        -------
        tuple
            ``(atomic_numbers, charges, positions, energy)`` as returned by
            OpenQDC: positions in Angstrom, energy in eV.
        """
        entry = self.ds[int(i)]
        return (
            np.asarray(entry.atomic_numbers, dtype=int),
            np.asarray(entry.charges, dtype=float),
            np.asarray(entry.positions, dtype=float),
            float(entry[target][method]),
        )

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


def inspect_dataset(name: str, cache_dir: str, download: bool = False) -> dict:
    """Collect metadata used to write a dataset config.

    The number of energy columns needs no data. Elements and size need the
    dataset to be downloaded.

    Parameters
    ----------
    name : str
        Dataset name.
    cache_dir : str
        OpenQDC cache directory.
    download : bool
        Download the dataset if it is not cached.

    Returns
    -------
    dict
        Metadata, JSON serialisable.
    """
    name = canonical_name(name)
    method_names = method_names_for(name)
    info = {
        "name": name,
        "n_methods": len(method_names),
        "method_names": method_names,
        "unusable": UNUSABLE.get(name),
        "not_total": NOT_TOTAL_ENERGY.get(name),
        "cached": False,
    }
    if info["unusable"]:
        return info
    if not (is_cached(name, cache_dir) or download):
        return info
    src = OpenQDCSource(name, cache_dir)
    info.update(
        cached=True,
        n_molecules=len(src),
        n_methods=src.n_methods,
        elements=src.elements(),
        max_atoms=int(src.ds.data["n_atoms"].max()),
    )
    return info
