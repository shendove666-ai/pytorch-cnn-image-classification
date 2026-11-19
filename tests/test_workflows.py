from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

from src import dataset, evaluate, experiments, reporting, train, utils


ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


class IndexedCifar(Dataset):
    """Make sample ordering observable without downloading CIFAR-10."""

    def __init__(self, root, train, download, transform):
        self.train = train
        self.transform = transform

    def __len__(self):
        return 128 if self.train else 80

    def __getitem__(self, index):
        return torch.full((3, 32, 32), float(index)), index % 10


@pytest.fixture
def indexed_cifar(monkeypatch):
    monkeypatch.setattr(dataset.datasets, "CIFAR10", IndexedCifar)


def first_image_indices(loader):
    images, _ = next(iter(loader))
    return images[:, 0, 0, 0].to(torch.int64).tolist()


def test_preview_advances_the_archived_training_sampler(indexed_cifar, tmp_path):
    baseline = dataset.get_dataloaders(batch_size=8, seed=42)
    augmented = dataset.get_dataloaders(batch_size=8, seed=42, use_augmentation=True)
    training = augmented["train"].dataset
    validation = augmented["val"].dataset
    official_test = augmented["test"].dataset
    assert training.indices == baseline["train"].dataset.indices
    assert validation.indices == baseline["val"].dataset.indices
    assert set(training.indices).isdisjoint(validation.indices)
    assert len(training) + len(validation) == 128
    assert training.dataset.train and validation.dataset.train and not official_test.train
    assert any(isinstance(step, transforms.RandomCrop) for step in training.dataset.transform.transforms)
    assert not any(isinstance(step, transforms.RandomCrop) for step in validation.dataset.transform.transforms)
    assert validation.dataset.transform is official_test.transform

    sampler_state = baseline["train"].generator.get_state().clone()
    preview_path = tmp_path / "preview.png"
    dataset.save_sample_grid(baseline["train"], preview_path)
    assert preview_path.exists()
    assert not torch.equal(baseline["train"].generator.get_state(), sampler_state)
    assert first_image_indices(baseline["train"]) != first_image_indices(augmented["train"])


def test_small_splits_and_smoke_subset_edges(indexed_cifar):
    no_validation = dataset.get_dataloaders(val_split=1e-9, seed=42)
    assert len(no_validation["val"].dataset) == 0
    assert train.validate_one_epoch(
        nn.Identity(), no_validation["val"], nn.CrossEntropyLoss(), CPU
    ) == (0.0, 0.0)
    with pytest.raises(StopIteration):
        dataset.inspect_batch(no_validation["val"])
    with pytest.raises(ValueError, match="num_samples"):
        dataset.get_dataloaders(smoke_test=True, smoke_subset_size=0)
    negative_slice = dataset.get_dataloaders(smoke_test=True, smoke_subset_size=-1)
    assert len(negative_slice["train"].dataset) == 115


def two_class_loader():
    images = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
    labels = torch.tensor([0, 1, 0, 1, 0])
    return DataLoader(TensorDataset(images, labels), batch_size=3)


def test_training_clears_gradients_and_validation_keeps_buffers():
    loader = two_class_loader()
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    with (
        patch.object(optimizer, "zero_grad", wraps=optimizer.zero_grad) as clear,
        patch.object(optimizer, "step", wraps=optimizer.step) as step,
    ):
        train.train_one_epoch(model, loader, criterion, optimizer, CPU)
    assert clear.call_count == step.call_count == 2

    before_validation = {key: value.clone() for key, value in model.state_dict().items()}
    loss, accuracy = train.validate_one_epoch(model, loader, criterion, CPU)
    assert not model.training
    assert all(torch.equal(value, before_validation[key]) for key, value in model.state_dict().items())
    with torch.no_grad():
        images, labels = loader.dataset.tensors
        logits = model(images)
        assert loss == pytest.approx(criterion(logits, labels).item())
        assert accuracy == pytest.approx((logits.argmax(1) == labels).float().mean().item())


def test_empty_validation_can_still_select_the_first_checkpoint(tmp_path):
    loader = two_class_loader()
    validation = DataLoader(
        TensorDataset(torch.empty(0, 3), torch.empty(0, dtype=torch.long)), batch_size=3
    )
    result = train.fit_model(nn.Linear(3, 2), loader, validation, 1, 0.01, 0.9, 0.0005,
                             CPU, tmp_path / "empty_validation.pth")
    assert result["best_epoch"] == 1
    assert result["best_val_acc"] == 0.0
    assert (tmp_path / "empty_validation.pth").is_file()


def test_nonfinite_validation_loss_can_coexist_with_perfect_accuracy():
    class NonfiniteLogits(nn.Module):
        def forward(self, images):
            return torch.tensor([[float("nan"), 1.0]] * len(images))

    loader = DataLoader(TensorDataset(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)))
    loss, accuracy = train.validate_one_epoch(NonfiniteLogits(), loader, nn.CrossEntropyLoss(), CPU)
    assert np.isnan(loss)
    assert accuracy == 1.0


@pytest.mark.parametrize("old_checkpoint_exists", [False, True])
def test_interrupt_before_first_epoch_preserves_checkpoint_path_contents(
    monkeypatch, tmp_path, old_checkpoint_exists
):
    checkpoint = tmp_path / "selected.pth"
    original_contents = b"previous result fixture"
    if old_checkpoint_exists:
        checkpoint.write_bytes(original_contents)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train, "train_one_epoch", interrupt)
    loader = two_class_loader()
    result = train.fit_model(nn.Linear(3, 2), loader, loader, 2, 0.01, 0.9, 0.0005, CPU, checkpoint)
    assert result["best_epoch"] == 0
    assert result["history"] == []
    assert checkpoint.exists() == old_checkpoint_exists
    if old_checkpoint_exists:
        assert checkpoint.read_bytes() == original_contents


def test_experiment_loads_selected_weights_before_touching_test_loader(monkeypatch, tmp_path):
    dataloaders = {name: object() for name in ("train", "val", "test")}
    model = nn.Linear(1, 10)
    checkpoint_state = {name: torch.full_like(value, 11.0) for name, value in model.state_dict().items()}
    events = []

    def fit(*, model, train_loader, val_loader, **kwargs):
        assert train_loader is dataloaders["train"] and val_loader is dataloaders["val"]
        events.append("fit")
        return {"history": [{}], "best_epoch": 1, "best_val_acc": 0.9, "total_training_time_sec": 0.5}

    def load(path, **kwargs):
        assert kwargs == {"map_location": CPU, "weights_only": True}
        events.append("load")
        return {"model_state_dict": checkpoint_state}

    def test_evaluation(selected_model, test_loader, device):
        assert test_loader is dataloaders["test"]
        assert device == CPU
        assert all(torch.equal(value, checkpoint_state[name])
                   for name, value in selected_model.state_dict().items())
        events.append("test")
        labels = list(range(10))
        return {**evaluate.compute_classification_metrics(labels, labels),
                "test_loss": 0.1, "correct_examples": [], "incorrect_examples": []}

    monkeypatch.setattr(experiments, "get_dataloaders", lambda **kwargs: dataloaders)
    monkeypatch.setattr(experiments, "get_device", lambda **kwargs: CPU)
    monkeypatch.setattr(experiments, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(experiments, "fit_model", fit)
    monkeypatch.setattr(experiments.torch, "load", load)
    monkeypatch.setattr(experiments, "evaluate_model", test_evaluation)
    for plot in ("plot_training_curves", "plot_confusion_matrix", "plot_prediction_examples"):
        monkeypatch.setattr(experiments, plot, lambda *args, **kwargs: None)

    config = utils.load_config(ROOT / "configs" / "default.yaml")
    config["training"]["save_dir"] = str(tmp_path / "metrics")
    config["training"]["figures_dir"] = str(tmp_path / "plots")
    logger = utils.setup_logger("tests.experiment_checkpoint")
    result = experiments.run_single_experiment(experiments.EXPERIMENT_DEFINITIONS[1], config, logger)
    assert events == ["fit", "load", "test"]
    assert result["test_acc"] == 1.0


def test_metric_column_input_exposes_existing_broadcast_boundary():
    labels = np.arange(3)
    metrics = evaluate.compute_classification_metrics(labels[:, None], labels, ["a", "b", "c"])
    assert metrics["test_accuracy"] == 0.333333
    assert np.trace(metrics["confusion_matrix"]) == 3


def test_cuda_availability_is_distinct_from_cpu_preference(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    assert utils.get_device().type == "cpu"
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.cuda, "get_device_name", lambda index: "CUDA fixture")
    assert utils.get_device().type == "cuda"
    assert utils.get_device(prefer_cuda=False).type == "cpu"
    assert utils.collect_environment_info()["active_device"] == "cuda"


def test_archived_seed_rows_and_aggregates_agree():
    archive = ROOT / "outputs" / "repeated"
    rows = pd.read_csv(archive / "all_seed_results.csv")
    summary = pd.read_csv(archive / "aggregate_summary.csv")
    assert len(rows) == 9
    assert set(rows["seed"]) == {41, 42, 43}
    assert (rows.groupby("seed")["exp_id"].nunique() == 3).all()
    assert (rows["epochs_run"] == 30).all()
    assert not rows["smoke_test"].any()
    assert np.isfinite(rows[list(reporting.REPEATED_METRICS)]).all().all()
    pd.testing.assert_frame_equal(
        reporting.summarize_repeated_results(rows), summary, check_exact=False, rtol=1e-12, atol=1e-12
    )


def test_yaml_and_named_logger_keep_their_current_boundaries(tmp_path):
    config = tmp_path / "configuration.yaml"
    config.write_text("", encoding="utf-8")
    assert utils.load_config(config) is None
    config.write_text("- not a mapping\n", encoding="utf-8")
    assert utils.load_config(config) == ["not a mapping"]

    old_log, new_log = tmp_path / "first.log", tmp_path / "second.log"
    logger = utils.setup_logger(f"tests.{tmp_path.name}", str(old_log))
    try:
        assert utils.setup_logger(logger.name, str(new_log)) is logger
        assert old_log.is_file() and not new_log.exists()
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
