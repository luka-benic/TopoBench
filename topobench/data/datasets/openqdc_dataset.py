"""Dataset class for energy-only OpenQDC datasets (QM9, Alchemy, SPICE, ...)."""

import json
import logging
import os
import os.path as osp

import numpy as np
import torch
from ase.data import covalent_radii
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import fs
from tqdm import tqdm

from topobench.data.utils.openqdc_utils import (
    OpenQDCSource,
    locked,
    resolve_energy_method,
    split_positions,
)

log = logging.getLogger(__name__)


class OpenQDCDataset(InMemoryDataset):
    r"""Energy-only OpenQDC dataset converted to PyG molecular graphs.

    Each conformer becomes one graph with:

    - ``x``: one-hot atomic number over ``atom_types`` + formal atomic charge,
    - ``pos``: Cartesian coordinates (Angstrom),
    - ``z``: atomic numbers,
    - ``edge_index``: covalent bonds inferred from interatomic distances
      (``edge_method="bonds"``) or all pairs within ``cutoff`` (``"radius"``),
    - ``openqdc_idx`` and ``n_atoms``: the OpenQDC molecule index and the
      number of atoms (used to identify molecules in the test-set results),
    - ``energy``: the selected energy in eV, as returned by OpenQDC,
    - ``y``: the regression target, i.e. ``energy`` minus a per-element
      linear baseline when ``baseline="per_element"``, else ``energy``.

    Forces are never loaded. Energies (eV) and positions (Angstrom) are the
    values OpenQDC returns for ``energy_unit="ev"``, ``distance_unit="ang"``
    (see :class:`~topobench.data.utils.openqdc_utils.OpenQDCSource`).

    Parameters
    ----------
    root : str
        Root directory for the processed PyG data.
    name : str
        OpenQDC class name, e.g. ``"QM9"``.
    parameters : DictConfig
        Loader parameters: ``openqdc_cache_dir``, ``atom_types``, ``target``,
        ``energy_method``, ``baseline``, ``edge_method``, ``bond_scale``,
        ``cutoff``, ``max_samples``, ``subsample_seed``.
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
            """Read an optional loader parameter.

            Parameters
            ----------
            key : str
                Parameter name.
            default : Any
                Value used when the parameter is missing.

            Returns
            -------
            Any
                The configured value or ``default``.
            """
            return OmegaConf.select(parameters, key, default=default)

        self.atom_types = [int(z) for z in parameters.atom_types]
        self.target = opt("target", "energies")
        self.energy_method = opt("energy_method", 0)
        self.baseline = opt("baseline", "per_element")
        self.edge_method = opt("edge_method", "bonds")
        self.bond_scale = float(opt("bond_scale", 1.3))
        self.cutoff = float(opt("cutoff", 5.0))
        self.max_samples = opt("max_samples")
        self.subsample_seed = int(opt("subsample_seed", 0))
        self.valid_fraction = float(opt("valid_fraction", 0.1))
        self.test_fraction = float(opt("test_fraction", 0.1))
        self.split_seed = int(opt("split_seed", 0))
        if self.target not in self.TARGETS:
            raise ValueError(f"target must be one of {self.TARGETS}.")
        if self.edge_method not in ("bonds", "radius"):
            raise ValueError("edge_method must be 'bonds' or 'radius'.")
        if self.baseline not in ("per_element", "none"):
            raise ValueError("baseline must be 'per_element' or 'none'.")
        processed_root = osp.join(root, name, self._tag())
        os.makedirs(processed_root, exist_ok=True)
        # Convert once: jobs started together wait here, then find data.pt.
        self.was_processed = False  # set by process() if it had to run
        with locked(
            osp.join(processed_root, "convert.lock"),
            f"{name} graphs (conversion)",
        ):
            super().__init__(root)

        out = fs.torch_load(self.processed_paths[0])
        data, self.slices, self.sizes, data_cls = out
        self.data = data_cls.from_dict(data)

        # The split is the one the MACE pipeline uses (``split_molecules``):
        # it depends on the OpenQDC indices only, not on the converted graphs.
        self.split_idx = {
            k: torch.as_tensor(v)
            for k, v in split_positions(
                self._data.openqdc_idx.numpy(),
                self.valid_fraction,
                self.test_fraction,
                self.split_seed,
            ).items()
        }

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

    def _tag(self) -> str:
        """Return a folder name unique to the conversion settings.

        Returns
        -------
        str
            The folder name.
        """
        edges = (
            f"bonds{self.bond_scale}"
            if self.edge_method == "bonds"
            else f"radius{self.cutoff}"
        )
        return "_".join(
            [
                self.target,
                f"m{self.energy_method}",
                # per-atom targets use atom fractions in the baseline
                f"b{self.baseline}"
                + (
                    "frac"
                    if self.target == "per_atom_formation_energies"
                    else ""
                ),
                edges,
                f"n{self.max_samples if self.max_samples is not None else 'all'}",
                f"s{self.subsample_seed}",
            ]
        )

    @property
    def processed_dir(self) -> str:
        """Return a processed directory unique to the conversion settings.

        Returns
        -------
        str
            Path to the processed directory.
        """
        self.processed_root = osp.join(self.root, self.name, self._tag())
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
        self.was_processed = True
        src = OpenQDCSource(self.name, self.raw_dir)
        names = src.method_names
        method = resolve_energy_method(names, self.energy_method)
        log.info("%s: energy column %d = %r", self.name, method, names[method])
        if src.not_total_reason and self.baseline != "none":
            log.warning(
                "%s energies are not total energies (%s); baseline=per_element "
                "is meaningless here (use baseline: none).",
                self.name,
                src.not_total_reason,
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
                    # Not named "...index": PyG would offset such keys when
                    # batching graphs.
                    openqdc_idx=torch.tensor([int(i)]),
                    n_atoms=torch.tensor([len(z)]),
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
            # A per-atom target is an intensive quantity: its regressors are
            # the atom fractions, not the atom counts.
            regressors = counts
            if self.target == "per_atom_formation_energies":
                regressors = counts / counts.sum(axis=1, keepdims=True)
            e0 = np.linalg.lstsq(regressors, energies, rcond=None)[0]
            counts = regressors
        residual = energies - counts @ e0
        for d, energy, y in zip(data_list, energies, residual, strict=True):
            d.energy = torch.tensor([energy], dtype=torch.float64)
            d.y = torch.tensor([y], dtype=torch.float32)

        with open(osp.join(self.processed_root, "baseline.json"), "w") as f:
            json.dump(
                {
                    "unit": "ev",
                    "baseline": self.baseline,
                    "atom_types": self.atom_types,
                    "e0": e0.tolist(),
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
