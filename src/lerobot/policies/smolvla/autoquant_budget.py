"""Scoped AutoQuantize budgets for fixed high-precision modules on ModelOpt 0.45.

The old solver includes disabled modules in its cost. Keeping their cost constant
lets us translate a scoped budget exactly, without changing ModelOpt on disk.
"""

import fnmatch
import logging
import math
from unittest.mock import patch


def translate_budget(target_bits: float, total_weights: int, excluded_weights: int) -> float:
    """Translate a budget excluding fixed 16-bit weights into the old full budget."""
    if not math.isfinite(target_bits) or not 0 < target_bits <= 16:
        raise ValueError("target_bits must be finite and in (0, 16].")
    if not 0 <= excluded_weights < total_weights:
        raise ValueError("Budget exclusions must leave at least one counted weight.")
    return (target_bits * (total_weights - excluded_weights) + 16 * excluded_weights) / total_weights


def scoped_bits(full_bits: float, total_weights: int, excluded_weights: int) -> float:
    return (full_bits * total_weights - 16 * excluded_weights) / (total_weights - excluded_weights)


def validate_exclusions(decisions, patterns):
    """Budget exclusion must not silently disable an eligible Linear."""
    names = set()
    for pattern in patterns:
        matches = [d for d in decisions if fnmatch.fnmatchcase(d.name, pattern)]
        if not matches:
            raise ValueError(f"cost_excluded_layers pattern matches no Linear: {pattern!r}")
        for decision in matches:
            if decision.eligible:
                raise ValueError(
                    f"Budget-excluded module {decision.name!r} is still eligible for quantization. "
                    "Keep it high precision using the existing exclusion settings first."
                )
            names.add(decision.name)
    return names


def auto_quantize_with_budget(mtq, model, *, cost_excluded_layers, **kwargs):
    """Run the existing gradient search with an equivalent budget.

Only fixed no-quant candidates may be excluded. Actual converted module sizes
are used, including Conv modules, rather than assuming all costs are Linear.
Raw ModelOpt logs/state retain the full-budget convention; the attached
``smolvla_budget`` report explicitly records both conventions.
"""
    if not cost_excluded_layers:
        return mtq.auto_quantize(model, **kwargs)

    from modelopt.torch.quantization import model_quant

    if kwargs.get("method", "gradient") != "gradient":
        raise ValueError("Scoped SmolVLA budgets currently support gradient search only.")
    constraints = kwargs["constraints"]
    if set(constraints) != {"effective_bits"}:
        raise ValueError("Scoped SmolVLA budgets require the default weight cost model.")
    target = constraints["effective_bits"]
    report = {}
    original_searcher = model_quant.AutoQuantizeGradientSearcher

    class ScopedBudgetSearcher(original_searcher):
        def run_search(self):
            modules = {
                name: module for name, module in self.model.named_modules()
                if self._is_auto_quantize_module(module)
            }
            excluded = {}
            for pattern in cost_excluded_layers:
                matches = {name: module for name, module in modules.items()
                           if fnmatch.fnmatchcase(name, pattern)}
                if not matches:
                    raise ValueError(f"Budget pattern matches no ModelOpt cost modules: {pattern!r}")
                excluded.update(matches)
            for name, module in excluded.items():
                choices = module.get_hparam("quant_recipe").choices
                if len(choices) != 1 or choices[0].compression != 1.0:
                    raise ValueError(f"Budget-excluded module must be fixed at 16 bits: {name}")
            total = sum(module.weight.numel() for module in modules.values())
            excluded_count = sum(module.weight.numel() for module in excluded.values())
            translated = translate_budget(target, total, excluded_count)
            report.update(
                implementation="fixed_bf16_budget_translation",
                cost_excluded_layers=list(cost_excluded_layers),
                excluded_module_names=sorted(excluded),
                total_weight_elements=total,
                excluded_weight_elements=excluded_count,
                counted_weight_elements=total - excluded_count,
                effective_bits_target=target,
                modelopt_effective_bits_target=translated,
            )
            logging.info(
                "AutoQuantize budget: scoped target=%.6f, equivalent full target=%.6f; "
                "excluded %d/%d weights (fixed BF16).", target, translated, excluded_count, total,
            )
            self.constraints = {**self.constraints, "effective_bits": translated}
            super().run_search()
            actual_full = self.best["constraints"]["effective_bits"]
            actual_scoped = scoped_bits(actual_full, total, excluded_count)
            report.update(
                modelopt_effective_bits_actual=actual_full,
                effective_bits_actual=actual_scoped,
                is_satisfied=bool(self.best["is_satisfied"] and actual_scoped <= target + 1e-6),
            )
            if not report["is_satisfied"]:
                raise RuntimeError("AutoQuantize did not satisfy the scoped bit budget.")

    # This CLI runs one search at a time. Restore the factory even if search fails.
    with patch.object(model_quant, "AutoQuantizeGradientSearcher", ScopedBudgetSearcher):
        quantized_model, state = mtq.auto_quantize(model, **kwargs)
    state["smolvla_budget"] = report
    logging.info("AutoQuantize scoped effective bits: %.6f", report["effective_bits_actual"])
    return quantized_model, state
