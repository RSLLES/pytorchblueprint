# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MetricCollection whose keys are created lazily via a factory."""

from collections.abc import Callable, Mapping
from typing import Any

from torchmetrics import Metric, MetricCollection


class LazyMetricCollection(MetricCollection):
    """A MetricCollection whose keys appear on first update."""

    def __init__(self, metric_factory: Callable[[], Metric]) -> None:
        super().__init__({}, compute_groups=False)
        self._metric_factory = metric_factory

    def update(self, values: Mapping[str, Any]) -> None:  # ty:ignore[invalid-method-override]  # noqa: D102
        for key, value in values.items():
            if key not in self:
                self.add_metrics({key: self._metric_factory()})
            self[key].update(value)
