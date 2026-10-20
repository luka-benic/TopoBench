"""Loader for energy-only OpenQDC datasets."""

from omegaconf import DictConfig

from topobench.data.datasets.openqdc_dataset import OpenQDCDataset
from topobench.data.loaders.base import AbstractLoader


class OpenQDCDatasetLoader(AbstractLoader):
    """Load an OpenQDC dataset (energies only) as PyG molecular graphs.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for the processed PyG data
            - data_name: OpenQDC class name (e.g. "Alchemy")
            - openqdc_cache_dir: Where OpenQDC downloads its arrays
            - atom_types: Atomic numbers used for the one-hot node features
            - target, energy_method, energy_unit, edge_method, bond_scale,
              cutoff, max_samples, subsample_seed: see ``OpenQDCDataset``.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> OpenQDCDataset:
        """Load the OpenQDC dataset.

        Returns
        -------
        OpenQDCDataset
            The converted dataset.
        """
        dataset = OpenQDCDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            parameters=self.parameters,
        )
        self.data_dir = dataset.processed_root
        return dataset

    def get_data_dir(self) -> str:
        """Get the processed data directory.

        Returns
        -------
        str
            The path to the dataset directory.
        """
        return self.data_dir
