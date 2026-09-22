import pytest
import torch

from scripts.train import load_g_base_state_dict


def test_g_accepts_m1_l_base_checkpoint(tmp_path):
    path = tmp_path / "M1-L_best.pt"
    state = {"weight": torch.tensor([1.0])}
    torch.save({"experiment": "M1-L", "model_state": state}, path)

    loaded = load_g_base_state_dict(path, torch.device("cpu"))

    assert torch.equal(loaded["weight"], state["weight"])


@pytest.mark.parametrize("experiment", ["M1", "B1", None])
def test_g_rejects_wrong_or_missing_base_experiment(tmp_path, experiment):
    path = tmp_path / "base.pt"
    checkpoint = {"model_state": {"weight": torch.tensor([1.0])}}
    if experiment is not None:
        checkpoint["experiment"] = experiment
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="richiesto experiment=M1-L"):
        load_g_base_state_dict(path, torch.device("cpu"))


def test_g_rejects_base_without_model_state(tmp_path):
    path = tmp_path / "base.pt"
    torch.save({"experiment": "M1-L"}, path)

    with pytest.raises(ValueError, match="model_state"):
        load_g_base_state_dict(path, torch.device("cpu"))
