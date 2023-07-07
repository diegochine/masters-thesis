import gin
import numpy as np
from pyagents.agents import SACLag


class SACLagEMS(SACLag):

    def init(self, envs, env_config=None, min_memories=None, actions=None, *args, **kwargs):
        self.num_envs = getattr(envs, "num_envs", 1)
        assert self.num_envs == 1, 'not yet implemented'  # TODO implement
        if self._wandb_run is not None and env_config is not None:
            self._wandb_run.config.update(env_config)
        if min_memories is None:
            min_memories = self._memory.get_config()['size']
        s_t, _ = envs.reset()
        for _ in range(min_memories // self.num_envs):
            a_t = envs.action_space.sample()
            s_tp1, r_t, terminated, truncated, info = envs.step(a_t)
            if 'final_info' in info:
                # all_costs = [info['final_info'][i].get('constraint_violation', 0.0) for i in range(self.num_envs)]
                r_t = np.column_stack([r_t, info['final_info'][0]['constraint_violation']])
                self.remember(state=s_t,
                              action=a_t,
                              reward=r_t,
                              next_state=info['final_observation'][0].reshape(self.num_envs, -1),
                              done=terminated)
            else:
                r_t = np.column_stack([r_t, info['constraint_violation']])
                self.remember(state=s_t,
                              action=a_t,
                              reward=r_t,
                              next_state=s_tp1,
                              done=terminated)
            s_t = s_tp1
