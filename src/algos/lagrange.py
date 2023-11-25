from tensordict import TensorDictBase
from torch import nn


class LagrangeBase(nn.Module):
    """Abstract base class for Lagrangian multipliers.
    """

    def forward(self, **kwargs) -> float:
        """Computes lagrangian loss.
        """
        raise NotImplementedError()

    def get(self):
        """Returns the current value of the lagrangian multiplier."""
        raise NotImplementedError()

    def get_logs(self) -> TensorDictBase:
        """Returns a tdict with log information."""
        raise NotImplementedError()
