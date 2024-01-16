import torch
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import inv_softplus

from src.algos.lagrange import LagrangeBase


class NaiveLagrange(LagrangeBase):
    """Implementation of naive Lagrangian multiplier, the simplest method for updating the lagrangian multiplier(s)
        when using the Lagrangian method.
    """

    def __init__(self, initial_value: float, cost_limit: float, lr: float, *args, **kwargs):
        """Initializes the module.
        :param initial_value: Initial value of the lagrangian multiplier.
        :param cost_limit: The cost limit.
        :param lr: learning rate
        """
        super().__init__(*args, **kwargs)
        assert initial_value >= 0.
        # To enforce lambda > 0, we train a real parameter lambda_0 and use relu to map it to R^+.
        lag = torch.nn.Parameter(torch.tensor(initial_value, dtype=torch.float32))
        self.register_parameter('lag', lag)
        self.register_buffer('cost_limit', torch.tensor(cost_limit))
        self.proj = torch.nn.functional.relu
        self.optim = torch.optim.Adam([self.lag], lr=lr, eps=1e-5)

    def forward(self, tdict: TensorDictBase, cost_scale: float) -> float:
        """Computes lagrangian loss.
        :param tdict: TensorDict with key 'avg_violation' containing the constraint violation of the last rollout.
        """
        lagrangian_loss = -self.proj(self.lag) * (tdict['avg_violation']) * cost_scale
        lagrangian_loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return lagrangian_loss

    def get(self):
        """Returns the current value of the lagrangian multiplier."""
        return self.proj(self.lag).detach()

    def get_logs(self) -> TensorDictBase:
        """Returns a tdict with log information."""
        return TensorDict({'lagrangian': [self.get()]}, batch_size=1)
