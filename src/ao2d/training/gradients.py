from __future__ import annotations

from collections.abc import Iterable

import torch


def grad_norm(parameters: Iterable[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    norms = torch.stack([torch.linalg.vector_norm(g, ord=norm_type).to(device) for g in grads])
    return float(torch.linalg.vector_norm(norms, ord=norm_type).detach())
