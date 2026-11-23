import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.cnn import binary_metrics, class_weights, evaluate, fit, seed_everything


def test_pr_auc_and_average_precision_are_distinguished():
    metrics = binary_metrics([0, 0, 1, 1], [.1, .4, .35, .8])
    assert metrics["pr_auc"] == pytest.approx(19 / 24)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["prevalence"] == .5
    assert binary_metrics([0, 0], [.1, .2])["pr_auc"] is None
    with pytest.raises(ValueError, match="finite"):
        binary_metrics([0, 1], [.1, np.nan])


def test_weights_use_only_training_class_counts():
    torch.testing.assert_close(class_weights([0] * 9 + [1]), torch.tensor([10 / 18, 5.]))
    with pytest.raises(ValueError, match="Training"):
        class_weights([0, 0])


class TinyDataset(TensorDataset):
    def __init__(self, count):
        generator = torch.Generator().manual_seed(count)
        y = torch.arange(count) % 2
        x = torch.randn(count, 4, generator=generator) + y[:, None]
        super().__init__(x, y)
        self.df = pd.DataFrame(dict(label=y.numpy(), patient_id=["patient1"] * (count // 2) + ["patient2"] * (count - count // 2)))


def test_training_is_repeatable_and_checkpoint_matches_saved_predictions(tmp_path):
    train = TinyDataset(20)
    validation = TinyDataset(12)
    config = dict(device="cpu", seed=5, samples_per_epoch=8, batch_size=4,
                  validation_batch_size=5, workers=0, learning_rate=.01,
                  weight_decay=0., gradient_clip=5., log_every=0, epochs=2, patience=0)
    histories = []
    for name in ("first", "second"):
        seed_everything(config["seed"])
        fit(nn.Linear(4, 2), train, validation, tmp_path / name, config, {})
        checkpoint = torch.load(tmp_path / name / "best.pt", weights_only=True, map_location="cpu")
        model = nn.Linear(4, 2)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics, labels, probabilities = evaluate(model, DataLoader(validation, batch_size=5), torch.device("cpu"))
        assert metrics["windows"] == len(validation)  # No validation sampling.
        saved = np.load(tmp_path / name / "best_validation_predictions.npz")
        np.testing.assert_array_equal(saved["labels"], labels)
        np.testing.assert_allclose(saved["probabilities"], probabilities)
        assert metrics["pr_auc"] == checkpoint["best_validation_pr_auc"]
        import json
        history = json.loads((tmp_path / name / "history.json").read_text())
        assert all(row["training_windows"] == 8 for row in history)
        histories.append([{k: v for k, v in row.items() if k != "seconds"} for row in history])
    assert histories[0] == histories[1]


def test_fit_applies_mixed_masks_to_training_only(tmp_path):
    import json
    from pathlib import Path
    experiment = json.loads(Path("configs/experiment.json").read_text())
    train = TensorDataset(torch.ones(32, 18, 4), torch.arange(32) % 2)
    train.df = pd.DataFrame(dict(label=(torch.arange(32) % 2).numpy()))
    val = TensorDataset(torch.ones(8, 18, 4), torch.arange(8) % 2)
    val.df = pd.DataFrame(dict(label=(torch.arange(8) % 2).numpy()))
    observed = {True: [], False: []}
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(18 * 4, 2)
        def forward(self, x):
            observed[self.training].extend((x.abs().sum(2) == 0).sum(1).tolist())
            return self.fc(x.flatten(1))
    config = dict(device="cpu", seed=5, samples_per_epoch=32, batch_size=8,
                  validation_batch_size=4, workers=0, learning_rate=.01,
                  weight_decay=0., gradient_clip=5., log_every=0, epochs=1, patience=0,
                  channel_masking="mixed", experiment=experiment)
    seed_everything(5)
    fit(Model(), train, val, tmp_path / "masked", config, {})
    assert set(observed[True]) == {0, 5, 9, 14}
    assert set(observed[False]) == {0}


def test_trainable_front_end_precedes_backbone_and_round_trips(tmp_path):
    class Gain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.full((18, 1), 2.0))

        def forward(self, x):
            return x * self.gain

    from src.models.cnn import ShallowEEGCNN

    config = dict(n_temporal_filters=2, dropout=0.0)
    front_end = Gain()
    model = ShallowEEGCNN(**config, front_end=front_end).eval()
    x = torch.randn(2, 18, 1024)
    x[:, 0] = 0
    seen = []
    hook = model.backbone.register_forward_pre_hook(lambda _module, args: seen.append(args[0]))
    scores = model(x)
    hook.remove()
    assert scores.shape == (2, 2)
    torch.testing.assert_close(seen[0], 2 * x)
    assert seen[0].dtype == x.dtype and seen[0].device == x.device
    loss = torch.nn.functional.cross_entropy(scores, torch.tensor([0, 1]))
    loss.backward()
    assert front_end.gain.grad.abs().sum() > 0
    assert "front_end.gain" in dict(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.equal(front_end.gain, torch.full_like(front_end.gain, 2.0))

    path = tmp_path / "front_end.pt"
    torch.save(dict(config=config, model_state_dict=model.state_dict()), path)
    payload = torch.load(path, weights_only=True)
    restored = ShallowEEGCNN(**payload["config"], front_end=Gain()).eval()
    restored.load_state_dict(payload["model_state_dict"])
    torch.testing.assert_close(restored.front_end.gain, front_end.gain)
    torch.testing.assert_close(restored(x), model(x))
