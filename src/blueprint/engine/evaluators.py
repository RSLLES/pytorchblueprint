# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluators turn model outputs into validation metrics, one batch at a time."""

from lightning.fabric import Fabric
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics import CatMetric, Metric

from blueprint.metrics import WassersteinDistance


class Evaluator:
    """Adapt one or more torchmetrics to the validation loop's uniform interface."""

    # skipped when a validation batch lacks one of these keys (e.g. no ground truth)
    REQUIRED_KEYS: tuple[str, ...] = ()

    def __init__(self, *metrics: Metric):
        self.metrics = metrics

    def prepare(self, fabric: Fabric, dl: DataLoader, model: nn.Module) -> None:
        """Move the owned metrics onto the fabric device before evaluation."""
        for metric in self.metrics:
            metric.to(fabric.device)

    def reset(self) -> None:
        """Reset every owned metric."""
        for metric in self.metrics:
            metric.reset()

    def update(self, output: Tensor, batch: dict[str, Tensor]) -> None:
        """Feed one batch of model outputs into the owned metrics."""
        raise NotImplementedError

    def compute(self) -> dict[str, Tensor | Figure]:
        """Return the named metric values for this evaluator."""
        raise NotImplementedError


class WassersteinEvaluator(Evaluator):
    """Earth mover's distance between the predicted and target distributions."""

    REQUIRED_KEYS = ("targetdist",)

    def __init__(self):
        self.distance = WassersteinDistance(p=1.0)
        super().__init__(self.distance)

    def update(self, output: Tensor, batch: dict[str, Tensor]) -> None:  # noqa: D102
        self.distance.update(output, batch["targetdist"])

    def compute(self) -> dict[str, Tensor | Figure]:  # noqa: D102
        return {"earth_mover_dist": self.distance.compute()}


class ScatterPlotEvaluator(Evaluator):
    """Scatter the source, predicted and target 2D samples side by side."""

    REQUIRED_KEYS = ("targetdist",)

    def __init__(self):
        self.source, self.result, self.target = CatMetric(), CatMetric(), CatMetric()
        super().__init__(self.source, self.result, self.target)

    def update(self, output: Tensor, batch: dict[str, Tensor]) -> None:  # noqa: D102
        self.source.update(batch["sourcedist"])
        self.result.update(output)
        self.target.update(batch["targetdist"])

    def compute(self) -> dict[str, Tensor | Figure]:  # noqa: D102
        panels = [
            ("Source", self.source, "blue"),
            ("Results", self.result, "red"),
            ("Target", self.target, "orange"),
        ]
        # a bare Figure is not registered with pyplot, so nothing has to close it
        fig = Figure(figsize=(12, 4))
        for ax, (title, metric, color) in zip(fig.subplots(1, 3), panels):
            x = metric.compute().cpu().numpy()
            ax.scatter(x[:, 0], x[:, 1], color=color, alpha=0.5)
            ax.set_title(title)
            ax.set_aspect("equal")
        fig.tight_layout()
        return {"fig": fig}
