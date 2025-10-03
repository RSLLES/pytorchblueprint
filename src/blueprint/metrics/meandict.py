# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

from torchmetrics import MeanMetric


class MeanDictMetric:
    def __init__(self, device=None, dtype=None):
        self.metrics = defaultdict(lambda: MeanMetric().to(device=device, dtype=dtype))

    def update(self, metrics_dict: dict):
        for key, value in metrics_dict.items():
            self.metrics[key].update(value)

    def compute(self) -> dict:
        return {k: m.compute() for k, m in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
