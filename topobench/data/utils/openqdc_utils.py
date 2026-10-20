"""Shared helpers for loading OpenQDC datasets (TopoBench and MACE pipelines).

Every dataset is loaded through OpenQDC's own unit conversion
(``energy_unit="ev"``, ``distance_unit="ang"``); the values OpenQDC returns are
used as they are. Forces are never read.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Units requested from OpenQDC for every dataset.
ENERGY_UNIT = "ev"
DISTANCE_UNIT = "ang"

# Datasets whose energy columns are not total energies.
NOT_TOTAL_ENERGY = {
    "QM8": "columns are excitation energies (E1, E2 x CC2/PBE0/PBE0.1/CAM)"
}
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
