"""Dataset class for energy-only OpenQDC datasets (QM9, Alchemy, SPICE, ...)."""

import json
import logging
import os.path as osp

import numpy as np
import torch
from ase.data import covalent_radii
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from tqdm import tqdm

from topobench.data.utils.openqdc_utils import OpenQDCSource

log = logging.getLogger(__name__)

# Bump when the conversion changes so stale processed files are not reused.
PROCESSING_VERSION = 2


class OpenQDCDataset(InMemoryDataset):
    r"""Energy-only OpenQDC dataset converted to PyG molecular graphs.

    Each conformer becomes one graph with:

    - ``x``: one-hot atomic number over ``atom_types`` + formal atomic charge,
    - ``pos``: Cartesian coordinates (Angstrom),
    - ``z``: atomic numbers,
    - ``edge_index``: covalent bonds inferred from interatomic distances
      (``edge_method="bonds"``) or all pairs within ``cutoff`` (``"radius"``),
    - ``energy``: the selected energy (``energy_unit``),
    - ``y``: the regression target, i.e. ``energy`` minus a per-element
      linear baseline when ``baseline="per_element"``, else ``energy``.

    Forces are never loaded. OpenQDC energies are read through
    :class:`~topobench.data.utils.openqdc_utils.OpenQDCSource`, which corrects
    wrong unit labels.

    Parameters
    ----------
    root : str
        Root directory for the processed PyG data.
    name : str
        OpenQDC class name, e.g. ``"QM9"``.
    parameters : DictConfig
        Loader parameters: ``openqdc_cache_dir``, ``atom_types``, ``target``,
        ``energy_method``, ``energy_unit``, ``stored_energy_unit``,
        ``stored_distance_unit``, ``baseline``, ``edge_method``,
        ``bond_scale``, ``cutoff``, ``max_samples``, ``subsample_seed``.
    """

    TARGETS = (
        "formation_energies",
        "energies",
        "per_atom_formation_energies",
    )

    def __init__(self, root: str, name: str, parameters: DictConfig) -> None:
        self.name = name
        self.parameters = parameters

        def opt(key, default=None):
            return OmegaConf.select(parameters, key, default=default)

        self.atom_types = [int(z) for z in parameters.atom_types]
        self.target = opt("target", "formation_energies")
        self.energy_method = opt("energy_method", 0)
        self.energy_unit = opt("energy_unit", "ev")
        self.stored_energy_unit = opt("stored_energy_unit")
        self.stored_distance_unit = opt("stored_distance_unit")
        self.baseline = opt("baseline", "per_element")
        self.edge_method = opt("edge_method", "bonds")
        self.bond_scale = float(opt("bond_scale", 1.3))
        self.cutoff = float(opt("cutoff", 5.0))
        self.max_samples = opt("max_samples")
        self.subsample_seed = int(opt("subsample_seed", 0))
        if self.target not in self.TARGETS:
            raise ValueError(f"target must be one of {self.TARGETS}.")
        if self.edge_method not in ("bonds", "radius"):
            raise ValueError("edge_method must be 'bonds' or 'radius'.")
        if self.baseline not in ("per_element", "none"):
            raise ValueError("baseline must be 'per_element' or 'none'.")
        super().__init__(root)

        out = fs.torch_load(self.processed_paths[0])
        data, self.slices, self.sizes, data_cls = out
        self.data = data_cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"{self.name}(root={self.root}, target={self.target}, "
            f"method={self.energy_method}, edge_method={self.edge_method})"
        )

    @property
    def raw_dir(self) -> str:
        """Return the OpenQDC cache directory.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return str(self.parameters.openqdc_cache_dir)

    @property
    def processed_dir(self) -> str:
        """Return a processed directory unique to the conversion settings.

        Returns
        -------
        str
            Path to the processed directory.
        """
        edges = (
            f"bonds{self.bond_scale}"
            if self.edge_method == "bonds"
            else f"radius{self.cutoff}"
        )
        tag = "_".join(
            [
                f"v{PROCESSING_VERSION}",
                self.target,
                f"m{self.energy_method}",
                self.energy_unit.replace("/", "per"),
                f"su{str(self.stored_energy_unit).replace('/', 'per')}",
                f"sd{self.stored_distance_unit}",
                f"b{self.baseline}",
                edges,
                f"n{self.max_samples if self.max_samples is not None else 'all'}",
                f"s{self.subsample_seed}",
            ]
        )
        self.processed_root = osp.join(self.root, self.name, tag)
        return osp.join(self.processed_root, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """OpenQDC manages its own raw files.

        Returns
        -------
        list[str]
            Empty list.
        """
        return []

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"

    def download(self) -> None:
        """Download is handled by OpenQDC inside :meth:`process`."""

    def _edges(self, pos: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Build an undirected edge index for one molecule.

        Parameters
        ----------
        pos : torch.Tensor
            Atom positions, shape ``[N, 3]``.
        z : torch.Tensor
            Atomic numbers, shape ``[N]``.

        Returns
        -------
        torch.Tensor
            Edge index, shape ``[2, E]``, both directions included.
        """
        dist = torch.cdist(pos, pos)
        if self.edge_method == "bonds":
            r = torch.as_tensor(covalent_radii, dtype=pos.dtype)[z]
            adj = dist < self.bond_scale * (r[:, None] + r[None, :])
        else:
            adj = dist < self.cutoff
        adj.fill_diagonal_(False)
        return adj.nonzero().t().contiguous()

    def process(self) -> None:
        """Convert the OpenQDC conformers to PyG graphs and save them."""
        method = self.energy_method
        src = OpenQDCSource(
            self.name,
            self.raw_dir,
            energy_unit=self.energy_unit,
            distance_unit="ang",
            stored_energy_unit=self.stored_energy_unit,
            stored_distance_unit=self.stored_distance_unit,
            energy_method=method if isinstance(method, int) else 0,
        )
        if not isinstance(method, int):
            from topobench.data.utils.openqdc_utils import (
                resolve_energy_method,
            )

            method = resolve_energy_method(src.method_names, method)
        if not 0 <= method < src.n_methods:
            raise ValueError(
                f"energy_method must be in [0, {src.n_methods - 1}] "
                f"for {self.name}."
            )
        if src.energy_kind == "not_total" and self.baseline != "none":
            log.warning(
                "%s energies are not total energies; baseline=per_element is "
                "meaningless here (use baseline: none).",
                self.name,
            )

        idxs = np.arange(len(src))
        if self.max_samples is not None and self.max_samples < len(src):
            rng = np.random.default_rng(self.subsample_seed)
            idxs = np.sort(
                rng.choice(len(src), int(self.max_samples), replace=False)
            )
        elif len(idxs) > 500_000:
            log.warning(
                "Converting %d molecules in Python will be slow and use a lot "
                "of memory; consider dataset.loader.parameters.max_samples.",
                len(idxs),
            )

        z_to_col = {z: i for i, z in enumerate(self.atom_types)}
        data_list, counts, energies = [], [], []
        for i in tqdm(idxs, desc=f"Converting {self.name}"):
            numbers, charges, positions, energy = src.get(
                i, self.target, method
            )
            if not np.isfinite(energy):
                continue  # energy not computed with this method
            unknown = set(numbers.tolist()) - z_to_col.keys()
            if unknown:
                raise ValueError(
                    f"Elements {sorted(unknown)} missing from atom_types "
                    f"{self.atom_types}; regenerate the config with "
                    "scripts/openqdc/generate_configs.py --datasets "
                    f"{self.name} --download."
                )
            z = torch.as_tensor(numbers, dtype=torch.long)
            pos = torch.as_tensor(positions, dtype=torch.float32)
            one_hot = torch.zeros(len(z), len(self.atom_types))
            cols = [z_to_col[a] for a in numbers.tolist()]
            one_hot[torch.arange(len(z)), cols] = 1.0
            charge = torch.as_tensor(charges, dtype=torch.float32)
            data_list.append(
                Data(
                    x=torch.cat([one_hot, charge[:, None]], dim=1),
                    pos=pos,
                    z=z,
                    edge_index=self._edges(pos, z),
                    num_nodes=len(z),
                )
            )
            counts.append(one_hot.sum(0).numpy())
            energies.append(energy)

        counts = np.stack(counts)
        energies = np.asarray(energies, dtype=np.float64)
        e0 = np.zeros(len(self.atom_types))
        if self.baseline == "per_element":
            # Linear per-element reference energies; the same least-squares fit
            # MACE uses for E0s=average, here on all converted molecules.
            e0 = np.linalg.lstsq(counts, energies, rcond=None)[0]
        residual = energies - counts @ e0
        for d, energy, y in zip(data_list, energies, residual, strict=True):
            d.energy = torch.tensor([energy], dtype=torch.float64)
            d.y = torch.tensor([y], dtype=torch.float32)

        with open(osp.join(self.processed_root, "baseline.json"), "w") as f:
            json.dump(
                {
                    "unit": self.energy_unit,
                    "baseline": self.baseline,
                    "atom_types": self.atom_types,
                    "e0": e0.tolist(),
                    "stored_energy_unit": src.stored_energy_unit,
                    "stored_distance_unit": src.stored_distance_unit,
                    "n_molecules": len(data_list),
                },
                f,
                indent=2,
            )

        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )
