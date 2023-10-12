import torch
from tensordict import TensorDictBase, TensorDict
from torch import nn


class PIDLagrange(nn.Module):
    """Implementation of PID Lagrangian multiplier. For more info: <https://arxiv.org/abs/2007.03964>`_
    """

    def __init__(self,
                 kp: float = 0.05,
                 ki: float = 0.0005,
                 kd: float = 0.1,
                 d_delay: int = 1,
                 alpha_p: float = 0.0,
                 alpha_d: float = 0.0,
                 initial_value: float = 1.0,
                 cost_limit: float = 0.0,
                 proj: str = 'relu',
                 *args, **kwargs):
        """Initializes the module.
        :param kp: The proportional gain of the PID controller.
        :param ki: The integral gain of the PID controller.
        :param kd: The derivative gain of the PID controller.
        :param d_delay: The delay of the derivative term.
        :param alpha_p: The exponential moving average alpha of the delta_p.
        :param alpha_d: The exponential moving average alpha of the delta_d.
        :param lagrangian_multiplier_init: The initial value of the lagrangian multiplier.
        :param cost_limit: The cost limit.
        :param proj: The projection function, either relu or softplus.
        """
        super().__init__(*args, **kwargs)
        self.register_buffer('kp', torch.tensor(kp))
        self.register_buffer('ki', torch.tensor(ki))
        self.register_buffer('kd', torch.tensor(kd))
        self.register_buffer('d_delay', torch.tensor(d_delay))
        self.register_buffer('alpha_p', torch.tensor(alpha_p))
        self.register_buffer('alpha_d', torch.tensor(alpha_d))
        self.register_buffer('pid_i', torch.tensor(initial_value))
        self.register_buffer('cost_limit', torch.tensor(cost_limit))
        self.register_buffer('prev_costs', torch.zeros((d_delay,)))
        self.register_buffer('delta_p', torch.tensor(0.0))
        self.register_buffer('cost_d', torch.tensor(0.0))
        self.register_buffer('lag', torch.tensor(0.0))

        assert proj in ('relu', 'softplus'), f"proj must be either 'relu' or 'softplus', got: {proj}"
        self.proj = nn.ReLU() if proj == 'relu' else nn.Softplus()

    def forward(self, tdict: TensorDictBase, **kwargs) -> float:
        """Updates the PID controller. """
        avg_violation = tdict.get('avg_violation').mean()
        delta = avg_violation - self.cost_limit
        self.pid_i = self.proj(self.pid_i + delta)

        self.delta_p = self.alpha_p * self.delta_p + (1 - self.alpha_p) * delta
        self.cost_d = self.alpha_d * self.cost_d + (1 - self.alpha_d) * avg_violation
        pid_d = self.proj(self.cost_d - self.prev_costs[0])
        pid_o = self.proj(self.kp * self.delta_p + self.ki * self.pid_i + self.kd * pid_d)

        loss_lagrangian = self.lag - pid_o
        self.lag = pid_o
        self.prev_costs = torch.roll(self.prev_costs, -1)
        self.prev_costs[-1] = self.cost_d

        return loss_lagrangian

    def get(self):
        """Returns the current value of the lagrangian multiplier."""
        return self.lag

    def get_logs(self) -> TensorDictBase:
        """Returns a tdict with log information."""
        return TensorDict({'lagrangian': [self.get()],
                           'pid_i': [self.pid_i],
                           'pid_p': [self.delta_p],
                           'pid_d': [self.proj(self.cost_d - self.prev_costs[0])],
                           'cost_d': [self.cost_d]
                           }, batch_size=1)
