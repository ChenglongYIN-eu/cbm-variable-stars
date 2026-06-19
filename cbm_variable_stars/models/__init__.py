"""Model architectures and factory functions."""

from typing import Dict, Any

from cbm_variable_stars.models.cbm_hard import (
    HardCBM, HardCBM_Linear, HardCBM_Calibrated, EndToEndHardCBM,
)
from cbm_variable_stars.models.cbm_soft import SoftCBM
from cbm_variable_stars.models.cem import ConceptEmbeddingModel
from cbm_variable_stars.models.mlp_baseline import BaselineMLP

MODEL_REGISTRY: Dict[str, type] = {
    "hard_cbm":        HardCBM,
    "hard_cbm_linear": HardCBM_Linear,
    "hard_cbm_cal":    HardCBM_Calibrated,
    "e2e_hard_cbm":    EndToEndHardCBM,
    "soft_cbm":        SoftCBM,
    "cem":             ConceptEmbeddingModel,
    "mlp":             BaselineMLP,
}


def create_model(name: str, **kwargs: Any) -> Any:
    """Model factory function."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def get_model_summary(name: str, **kwargs: Any) -> str:
    """Return model parameter count summary."""
    model = create_model(name, **kwargs)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{name}: {total:,} trainable parameters"
