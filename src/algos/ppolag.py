import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential
from torchrl.objectives import distance_loss, ClipPPOLoss
from torchrl.objectives.value import ValueEstimatorBase


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
            *,
            target_kl: float = 0.02,
            reward_scale: float = 1.0,
            cost_scale: float = 1.0,
            beta_ema: float = 0.9,
            **kwargs,
    ):
        super(PPOLagLoss, self).__init__(actor, critic, **kwargs)
        self.convert_to_functional(safe_critic, 'safe_critic', create_target_params=False)
        self.r_value_estimator = r_value_estimator
        self.c_value_estimator = c_value_estimator
        self.register_buffer('lagrangian', torch.ones(1))
        self.register_buffer('cost_scale', torch.tensor(cost_scale))
        self.register_buffer('reward_scale', torch.tensor(reward_scale))
        self.register_buffer('target_kl', torch.tensor(target_kl))
        self.register_buffer('beta', torch.ones(1))
        self.register_buffer('beta_ema', torch.tensor(beta_ema))

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = super().out_keys()
            del keys['loss_critic']
            keys.extend(['loss_r_critic', 'loss_c_critic', 'loss_lagrangian'])
            self._out_keys = keys
        return self._out_keys

    def set_lagrangian(self, lagrangian: float):
        """Updates the costs of the lagrangian multiplier."""
        self.lagrangian = lagrangian

    def update_beta(self, tdict: TensorDictBase, optim: torch.optim.Optimizer, grad_clip_norm: float):
        """Compute adaptive scaling parameter beta as the ratio of un-scaled policy gradients.
        See PID lagrangian paper for more info (section 7)"""
        tmp_td = tdict.clone(False)

        # compute advantages for both critics
        r_tmp_td = tdict.clone(False).set(('next', 'reward'), tdict.get(('next', 'reward'))[:, :1] * self.reward_scale)
        c_tmp_td = tdict.clone(False).set(('next', 'reward'), tdict.get(('next', 'reward'))[:, 1:] * self.cost_scale)

        with torch.no_grad():
            self.r_value_estimator(
                r_tmp_td,
                params=self._cached_critic_params_detached,
                target_params=self.target_critic_params,
            )
            self.c_value_estimator(
                c_tmp_td,
                params=self.safe_critic_params.detach(),
                target_params=self.target_safe_critic_params,
            )
        r_advantage = r_tmp_td.get(self.r_value_estimator.tensor_keys.advantage)
        c_advantage = c_tmp_td.get(self.c_value_estimator.tensor_keys.advantage)

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
        pi_logratio, dist = self._log_weight(tmp_td)
        pi_ratio = pi_logratio.exp()
        clipped_ratio = pi_ratio.clamp(1. - self.clip_epsilon, 1. + self.clip_epsilon)
        # compute surrogate losses for both reward and cost
        r_gain1 = pi_ratio * r_advantage
        r_gain2 = clipped_ratio * r_advantage
        r_gain = - torch.stack([r_gain1, r_gain2], -1).min(dim=-1)[0]
        c_gain1 = pi_ratio * c_advantage
        c_gain2 = clipped_ratio * c_advantage
        c_gain = torch.stack([c_gain1, c_gain2], -1).max(dim=-1)[0]

        optim.zero_grad()
        r_gain.mean().backward(retain_graph=True)
        r_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
        r_grad_norm = torch.minimum(r_grad_norm, torch.tensor(grad_clip_norm))
        optim.zero_grad()
        c_gain.mean().backward()
        c_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
        c_grad_norm = torch.minimum(c_grad_norm, torch.tensor(grad_clip_norm))
        optim.zero_grad()

        self.beta = self.beta_ema * self.beta + ((1 - self.beta_ema) * r_grad_norm / c_grad_norm)

    def forward(self, tdict: TensorDictBase) -> TensorDictBase:
        tmp_td = tdict.clone(False)
        td_out = TensorDict({}, [])

        # compute advantages for both critics
        r_tmp_td = tdict.clone(False).set(('next', 'reward'), tdict.get(('next', 'reward'))[:, :1] * self.reward_scale)
        c_tmp_td = tdict.clone(False).set(('next', 'reward'), tdict.get(('next', 'reward'))[:, 1:] * self.cost_scale)

        with torch.no_grad():
            self.r_value_estimator(
                r_tmp_td,
                params=self._cached_critic_params_detached,
                target_params=self.target_critic_params,
            )
            self.c_value_estimator(
                c_tmp_td,
                params=self.safe_critic_params.detach(),
                target_params=self.target_safe_critic_params,
            )
        r_advantage = r_tmp_td.get(self.r_value_estimator.tensor_keys.advantage)
        c_advantage = c_tmp_td.get(self.c_value_estimator.tensor_keys.advantage)

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
        pi_logratio, dist = self._log_weight(tmp_td)
        pi_ratio = pi_logratio.exp()
        approx_kl = ((pi_ratio - 1) - pi_logratio).mean()  # kl estimator, see http://joschu.net/blog/kl-approx.html
        td_out.set("approx_kl", approx_kl)
        if approx_kl <= self.target_kl:  # early stopping if kl-divergence is too large
            clipped_ratio = pi_ratio.clamp(1. - self.clip_epsilon, 1. + self.clip_epsilon)
            # compute surrogate losses for both reward and cost
            r_gain1 = pi_ratio * r_advantage
            r_gain2 = clipped_ratio * r_advantage
            r_gain = torch.stack([r_gain1, r_gain2], -1).min(dim=-1)[0]
            c_gain1 = pi_ratio * c_advantage
            c_gain2 = clipped_ratio * c_advantage
            c_gain = torch.stack([c_gain1, c_gain2], -1).max(dim=-1)[0]

            loss_pi = (-r_gain + self.lagrangian * self.beta * c_gain).mean() / (1 + self.lagrangian)
            td_out.set("loss_pi", loss_pi)

        # ESS for logging
        with torch.no_grad():
            lw = pi_logratio.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = pi_logratio.shape[0]
            td_out.set('ESS', ess.mean() / batch)

        # compute entropy and entropy loss
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("loss_entropy", -self.entropy_coef * entropy.mean())
            td_out.set("entropy", entropy.mean().detach())

        # compute critics losses
        target = r_tmp_td.get(self.r_value_estimator.tensor_keys.value_target)
        pred = self.critic(r_tmp_td, params=self.critic_params).get(self.r_value_estimator.tensor_keys.value)
        loss_r_critic = self.critic_coef * distance_loss(
            target, pred,
            loss_function=self.loss_critic_type
        )
        td_out.set('preds_reward', pred.detach())
        td_out.set('targets_reward', target.detach())

        target = c_tmp_td.get(self.c_value_estimator.tensor_keys.value_target)
        pred = self.safe_critic(c_tmp_td, params=self.safe_critic_params).get(self.c_value_estimator.tensor_keys.value)
        loss_c_critic = self.critic_coef * distance_loss(
            target, pred,
            loss_function=self.loss_critic_type
        )
        td_out.set('preds_constraint', pred.detach())
        td_out.set('targets_constraint', target.detach())

        td_out.set("loss_r_critic", loss_r_critic.mean())
        td_out.set("loss_c_critic", loss_c_critic.mean())

        return td_out
