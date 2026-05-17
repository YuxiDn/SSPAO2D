"""Learning-rate scheduler helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, ReduceLROnPlateau, StepLR


def _get(config: Mapping[str, object], key: str, default: object = None, prefix: str = "") -> object:
    if prefix:
        prefixed = f"{key}_{prefix}"
        if prefixed in config:
            return config[prefixed]
    return config.get(key, default)


def _scheduler_name(config: Mapping[str, object], prefix: str = "") -> str:
    raw = _get(config, "scheduler", None, prefix)
    if raw is None:
        raw = _get(config, "lr_scheduler", "StepLR", prefix)
    name = str(raw).lower()
    aliases = {
        "steplr": "step",
        "cosineannealinglr": "cosine",
        "multisteplr": "multistep",
        "reducelronplateau": "plateau",
    }
    return aliases.get(name, name)


def build_scheduler(
    optimizer: Optimizer,
    training_config: Mapping[str, object],
    total_epochs: int,
    prefix: str = "",
):
    """Build a learning-rate scheduler from the training config.

    Supported schedulers are ``step``, ``cosine``, ``multistep``, ``plateau``,
    and ``none``. SSPAO-style names such as ``StepLR``,
    ``CosineAnnealingLR``, and ``ReduceLROnPlateau`` are also accepted.
    """

    name = _scheduler_name(training_config, prefix)
    if name in {"none", "off", "false", "null"}:
        return None

    min_lr = float(_get(training_config, "min_lr", _get(training_config, "lr_min", 0.0, prefix), prefix))
    gamma = float(_get(training_config, "lr_gamma", _get(training_config, "gamma", 0.5, prefix), prefix))

    if name == "cosine":
        t_max = int(_get(training_config, "T_max", total_epochs, prefix))
        return CosineAnnealingLR(optimizer, T_max=max(1, t_max), eta_min=min_lr)
    if name == "step":
        default_step = max(1, int(total_epochs) // 4)
        step_size = int(_get(training_config, "lr_step_size", _get(training_config, "step_size", default_step, prefix), prefix))
        return StepLR(optimizer, step_size=max(1, step_size), gamma=gamma)
    if name == "multistep":
        raw_milestones = _get(training_config, "lr_milestones", None, prefix)
        if raw_milestones is None:
            raw_milestones = [max(1, int(total_epochs) // 2), max(1, int(total_epochs) * 3 // 4)]
        if not isinstance(raw_milestones, Sequence) or isinstance(raw_milestones, (str, bytes)):
            raise ValueError("lr_milestones must be a list of epoch numbers.")
        milestones = sorted({int(m) for m in raw_milestones if int(m) > 0})
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    if name == "plateau":
        factor = float(_get(training_config, "lr_factor", _get(training_config, "factor", gamma, prefix), prefix))
        patience = int(_get(training_config, "lr_patience", _get(training_config, "patience", 10, prefix), prefix))
        threshold = float(_get(training_config, "threshold", 1e-4, prefix))
        mode = str(_get(training_config, "mode", "min", prefix))
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, min_lr=min_lr)

    raise ValueError(f"Unsupported scheduler: {name}")


def step_scheduler(scheduler, metric: float | None = None) -> None:
    """Advance a scheduler by one epoch."""

    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        if metric is None:
            raise ValueError("ReduceLROnPlateau requires a validation metric.")
        scheduler.step(float(metric))
    else:
        scheduler.step()


def get_current_lr(optimizer: Optimizer) -> float:
    """Return the first parameter group's learning rate."""

    return float(optimizer.param_groups[0]["lr"])
