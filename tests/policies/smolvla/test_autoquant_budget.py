"""Budget scope must change accounting, not the optimization feasible set."""
from itertools import product
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.autoquant_budget import (
    auto_quantize_with_budget, scoped_bits, translate_budget, validate_exclusions,
)


def test_budget_translation_preserves_all_feasible_assignments():
    sizes = [32, 64, 96]
    excluded = 128
    total = sum(sizes) + excluded
    for target in (4.5, 6, 8.5, 12, 16):
        full_target = translate_budget(target, total, excluded)
        for bits in product((4, 8, 16), repeat=len(sizes)):
            cost = sum(n * b for n, b in zip(sizes, bits))
            full = (cost + 16 * excluded) / total
            assert (cost <= target * sum(sizes)) == (full <= full_target)
            assert scoped_bits(full, total, excluded) == pytest.approx(cost / sum(sizes))


def test_exclusions_require_existing_high_precision_and_valid_names():
    decisions = [SimpleNamespace(name="vision.proj", eligible=False),
                 SimpleNamespace(name="expert.proj", eligible=True)]
    assert validate_exclusions(decisions, ["vision.*", "vision.proj"]) == {"vision.proj"}
    with pytest.raises(ValueError, match="matches no Linear"):
        validate_exclusions(decisions, ["visual.*"])
    with pytest.raises(ValueError, match="still eligible"):
        validate_exclusions(decisions, ["expert.*"])
    with pytest.raises(ValueError, match="leave at least"):
        translate_budget(8, 100, 100)


def test_empty_exclusions_delegate_unchanged():
    calls = []
    mtq = SimpleNamespace(auto_quantize=lambda *args, **kwargs: calls.append((args, kwargs)))
    model = object()
    auto_quantize_with_budget(mtq, model, cost_excluded_layers=[], constraints={"effective_bits": 8})
    assert calls == [((model,), {"constraints": {"effective_bits": 8}})]


def test_real_modelopt_search_keeps_excluded_weights_and_restores_factory():
    mtq = pytest.importorskip("modelopt.torch.quantization")
    from modelopt.torch.quantization import model_quant

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.Linear(16, 16, bias=False)
            self.expert = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.expert(self.vision(x)).square().mean()

    torch.manual_seed(3)
    model = TinyModel()
    original = model.vision.weight.detach().clone()
    factory = model_quant.AutoQuantizeGradientSearcher
    candidates = [
        {"quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": bits, "axis": None}},
        ], "algorithm": "max"}
        for bits in (4, 8)
    ]
    result, state = auto_quantize_with_budget(
        mtq, model, cost_excluded_layers=["vision"],
        constraints={"effective_bits": 8}, quantization_formats=candidates,
        disabled_layers=["vision"], data_loader=[torch.randn(2, 16)],
        forward_step=lambda m, x: m(x), loss_func=lambda output, batch: output,
        num_calib_steps=1, num_score_steps=1, method="gradient", verbose=False,
    )
    report = state["smolvla_budget"]
    assert report["modelopt_effective_bits_target"] == 12
    assert report["excluded_weight_elements"] == 256
    assert report["effective_bits_actual"] <= 8 + 1e-6
    assert report["is_satisfied"]
    assert result.vision.weight_quantizer._disabled
    assert torch.equal(result.vision.weight, original)
    assert model_quant.AutoQuantizeGradientSearcher is factory
