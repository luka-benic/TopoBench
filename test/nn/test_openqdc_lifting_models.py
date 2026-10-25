"""Small molecular runs through the OpenQDC conversion and training pipeline."""

from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from lightning import Trainer
from omegaconf import OmegaConf

from topobench.data.datasets import openqdc_dataset
from topobench.data.preprocessor.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader


class MolecularSource:
    """Supplied geometries with synthetic scalar targets, requiring no download."""

    method_names = ["energy"]
    not_total_reason = None

    def __init__(self, *args):
        pass

    def __len__(self):
        return 8

    def get(self, index, target, method):
        if index % 2:
            numbers = np.array([8, 1, 1])
            positions = np.array(
                [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.92, 0.0]]
            )
        else:
            numbers = np.array([6, 6, 7, 8])
            positions = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 0.87, 0.0],
                    [0.5, 0.29, 0.82],
                ]
            )
        return numbers, np.zeros(len(numbers)), positions, -1.0 - index / 10


@pytest.mark.parametrize(
    "model_name",
    [
        "cell/can",
        "cell/cccn",
        "cell/ccxn",
        "cell/cwn",
        "simplicial/scn",
        "simplicial/sccnn",
        "simplicial/sccnn_custom",
        "simplicial/san",
        "hypergraph/unignn",
        "hypergraph/unignn2",
        "graph/gcn_cluster",
    ],
)
def test_one_epoch_and_checkpoint_reload(model_name, tmp_path, monkeypatch):
    monkeypatch.setattr(openqdc_dataset, "OpenQDCSource", MolecularSource)
    monkeypatch.setattr(
        openqdc_dataset, "method_names_for", lambda name: ["energy"]
    )
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with hydra.initialize_config_dir(str(config_dir), version_base="1.3"):
        cfg = hydra.compose(
            config_name="run",
            overrides=[
                "dataset=graph/openqdc_qm9",
                f"model={model_name}",
                "trainer=cpu",
                f"paths.data_dir={tmp_path}",
                "dataset.loader.parameters.valid_fraction=0.25",
                "dataset.loader.parameters.test_fraction=0.25",
                "dataset.loader.parameters.edge_method=radius",
                "dataset.loader.parameters.cutoff=1.1",
            ],
        )
    torch.manual_seed(42)
    loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = loader.load()
    preprocessor = PreProcessor(
        dataset, dataset_dir, hydra.utils.instantiate(cfg.get("transforms"))
    )
    for original, lifted in zip(dataset, preprocessor, strict=True):
        for key in ["y", "energy", "pos", "z", "openqdc_idx"]:
            torch.testing.assert_close(original[key], lifted[key])
    splits = preprocessor.load_dataset_splits(cfg.dataset.split_params)
    datamodule = TBDataloader(*splits, batch_size=2, num_workers=0)
    model = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
    )
    trainer.fit(model, datamodule=datamodule)
    assert trainer.global_step == 2
    assert any(
        not torch.equal(initial[name], value)
        for name, value in model.named_parameters()
    )
    assert all(torch.isfinite(value).all() for value in model.parameters())
    scores = trainer.test(model, datamodule=datamodule, verbose=False)[0]
    assert np.isfinite(scores["test/mae"])
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    reloaded = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )
    reloaded.load_state_dict(
        torch.load(checkpoint, weights_only=False)["state_dict"]
    )
    model.eval()
    reloaded.eval()
    batch = next(iter(datamodule.test_dataloader()))
    with torch.no_grad():
        torch.testing.assert_close(
            model(batch.clone())["logits"], reloaded(batch.clone())["logits"]
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "cell/hopse_g",
        "cell/hopse_m",
        "cell/topotune",
        "cell/topotune_onehasse",
        "simplicial/hopse_g",
        "simplicial/hopse_m",
        "simplicial/topotune",
    ],
)
def test_neighborhood_requirements_are_resolved(model_name):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with hydra.initialize_config_dir(str(config_dir), version_base="1.3"):
        cfg = hydra.compose(
            config_name="run",
            overrides=["dataset=graph/openqdc_qm9", f"model={model_name}"],
        )
    domain = model_name.split("/")[0]
    lifting = cfg.transforms[f"graph2{domain}_lifting"]
    requirements = OmegaConf.select(cfg.model, "backbone.neighborhoods")
    if requirements is None:
        requirements = cfg.model.preprocessing_params.neighborhoods
    assert list(lifting.neighborhoods) == list(requirements)
