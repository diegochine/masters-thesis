import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential
from torchrl.objectives import distance_loss, ClipPPOLoss
from torchrl.objectives.value import ValueEstimatorBase

from src.agents.naive_lagrange import NaiveLagrange
from src.agents.pid_lagrange import PIDLagrange


class PPOLagLoss(ClipPPOLoss):
    """PPO Lag loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    """

    def __init__(
            self,
            actor: ProbabilisticTensorDictSequential,
            critic: TensorDictModule,
            safe_critic: TensorDictModule,
            r_value_estimator: ValueEstimatorBase,
            c_value_estimator: ValueEstimatorBase,
            lagrangian: NaiveLagrange | PIDLagrange,
            lagrangian_delay: int = 50,
            *,
            r_advantage_key: str = "r_advantage",
            c_advantage_key: str = "c_advantage",
            r_value_target_key: str = "r_value_target",
            c_value_target_key: str = "c_value_target",
            r_value_key: str = "r_value",
            c_value_key: str = "c_value",
            **kwargs,
    ):
        super(PPOLagLoss, self).__init__(actor, critic,
                                         advantage_key=r_advantage_key,
                                         value_key=r_value_key,
                                         value_target_key=r_value_target_key,
                                         **kwargs)
        self.lag = lagrangian
        self.convert_to_functional(safe_critic, 'safe_critic', create_target_params=False)
        self.c_advantage_key = c_advantage_key
        self.c_value_target_key = c_value_target_key
        self.c_value_key = c_value_key
        self.r_value_estimator = r_value_estimator
        self.c_value_estimator = c_value_estimator
        self.register_buffer('lagrangian_delay', torch.tensor(lagrangian_delay))
        self.register_buffer('step', torch.tensor(0))

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = super().out_keys()
            del keys['loss_critic']
            keys.extend(['loss_r_critic', 'loss_c_critic', 'loss_lagrangian'])
            self._out_keys = keys
        return self._out_keys

    def forward(self, tdict: TensorDictBase) -> TensorDictBase:
        tmp_td = tdict.clone(False)
        td_out = TensorDict({}, [])

        if self.step % self.lagrangian_delay == 0:
            # compute lagrangian loss
            loss_lagrangian = self.lag(tmp_td)
            td_out.set("loss_lagrangian", loss_lagrangian)
        self.step = (self.step + 1) % self.lagrangian_delay
        td_out.set("lagrangian", self.lag.get())

        # compute advantages for both critics
        tmp_td = tmp_td.set(('next', 'reward'), tdict.get(('next', 'reward'))[:, :1])
        self.r_value_estimator(
            tmp_td,
            params=self.critic_params.detach(),
            target_params=self.target_critic_params,
        )
        tmp_td = tmp_td.set(('next', 'reward'), tdict.get(('next', 'reward'))[:, 1:])
        self.c_value_estimator(
            tmp_td,
            params=self.safe_critic_params.detach(),
            target_params=self.target_safe_critic_params,
        )

        r_advantage = tmp_td.get(self.advantage_key)
        c_advantage = tmp_td.get(self.c_advantage_key)

        if self.normalize_advantage:
            if r_advantage.numel() > 1:
                loc = r_advantage.mean().item()
                scale = r_advantage.std().clamp_min(1e-6).item()
                r_advantage = (r_advantage - loc) / scale
            if c_advantage.numel() > 1:
                loc = c_advantage.mean().item()
                scale = c_advantage.std().clamp_min(1e-6).item()
                c_advantage = (c_advantage - loc) / scale

        # compute actor loss
        log_weight, dist = self._log_weight(tmp_td)
        assert c_advantage.shape == log_weight.shape and r_advantage.shape == log_weight.shape

        r_gain1 = log_weight.exp() * r_advantage
        r_gain2 = log_weight.clamp(*self._clip_bounds).exp() * r_advantage
        r_gain = torch.stack([r_gain1, r_gain2], -1).min(dim=-1)[0]
        c_gain = log_weight.exp() * c_advantage

        loss_pi = (-r_gain + self.lag.get() * c_gain).mean() / (1 + self.lag.get())
        td_out.set("loss_pi", loss_pi)

        # compute entropy and entropy loss
        entropy = self.get_entropy_bonus(dist)
        if self.entropy_bonus:
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
        td_out.set("entropy", entropy.mean().detach())

        # compute critics losses
        target = tmp_td.get(self.value_target_key)
        pred = self.critic(tmp_td, params=self.critic_params).get(self.value_key)
        loss_r_critic = self.critic_coef * distance_loss(
            target, pred,
            loss_function=self.loss_critic_type
        )
        td_out.set('preds_reward', pred.detach())
        td_out.set('targets_reward', target.detach())

        target = tmp_td.get(self.c_value_target_key)
        pred = self.safe_critic(tmp_td, params=self.safe_critic_params).get(self.c_value_key)
        loss_c_critic = self.critic_coef * distance_loss(
            target, pred,
            loss_function=self.loss_critic_type
        )
        td_out.set('preds_constraint', pred.detach())
        td_out.set('targets_constraint', target.detach())

        td_out.set("loss_r_critic", loss_r_critic.mean())
        td_out.set("loss_c_critic", loss_c_critic.mean())

        return td_out
