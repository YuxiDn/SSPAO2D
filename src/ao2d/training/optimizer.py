"""Optimizer helpers compatible with the original SSPAO config style."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch import Tensor
from torch.optim import Adam, AdamW, Optimizer, SGD


def get_learning_rate(training_config: Mapping[str, object], prefix: str = "") -> float:
    """Resolve a learning rate from either new or SSPAO-style config keys."""

    if prefix:
        for key in (f"lr_{prefix}", f"initial_learning_rate_{prefix}"):
            if key in training_config:
                return float(training_config[key])
    if "lr" in training_config:
        return float(training_config["lr"])
    return float(training_config.get("initial_learning_rate", 1e-4))


def build_optimizer(params: Iterable[Tensor], training_config: Mapping[str, object], prefix: str = "") -> Optimizer:
    """Build Adam, AdamW, or SGD from the training config.

    SSPAO commonly used Adam with ``initial_learning_rate``. AdamW remains
    available when explicitly requested.
    """

    name = str(training_config.get("optimizer", "Adam")).lower()
    lr = get_learning_rate(training_config, prefix=prefix)
    weight_decay = float(training_config.get("weight_decay", 0.0))

    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(training_config.get("sgd_momentum", training_config.get("momentum", 0.9)))
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {training_config.get('optimizer')}")


def set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
    """Set every parameter group's learning rate."""

    for group in optimizer.param_groups:
        group["lr"] = float(lr)
