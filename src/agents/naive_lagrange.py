import torch
from tensordict import TensorDictBase
from tensordict.nn import inv_softplus
from torch import nn


class NaiveLagrange(nn.Module):
    """Implementation of naive Lagrangian multiplier, the simplest method for updating the lagrangian multiplier(s)
        when using the Lagrangian method.
    """

    def __init__(self, initial_value: float, cost_limit: float, *args, **kwargs):
        """Initializes the module.
        :param initial_value: Initial value of the lagrangian multiplier.
        :param cost_limit: The cost limit.
        """
        super().__init__(*args, **kwargs)
        # To enforce lambda > 0, we train a real parameter lambda_0 and use softplus to map it to R^+.
        lag = torch.nn.Parameter(torch.tensor(inv_softplus(initial_value)))
        self.register_parameter('lag', lag)
        self.register_buffer('cost_limit', torch.tensor(cost_limit))
        self.proj = torch.nn.functional.softplus

    def forward(self, tdict: TensorDictBase):
        """Computes lagrangian loss.
        :param tdict: TensorDict with key 'avg_violation' containing the constraint violation of the last rollout.
        """
        lagrangian_loss = -self.proj(self.lag) * (tdict.get('avg_violation').mean() - self.cost_limit)
        # tdict.set('loss_lagrangian', lagrangian_loss)
        return lagrangian_loss

    def get(self):
        """Returns the current value of the lagrangian multiplier."""
        return torch.nn.functional.softplus(self.lag).detach()
