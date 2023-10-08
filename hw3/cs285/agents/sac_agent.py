from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
import cs285.infrastructure.sac_utils as sac_utils
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic

        ### my code starts here ###
        # alpha is the temp
        with torch.no_grad():
            action_distribution = self.actor(next_ob_no)
            sampled_actions = action_distribution.sample()
            targ_q_1, targ_q_2 = self.critic_target(next_ob_no, sampled_actions)

            log_prob = action_distribution.log_prob(sampled_actions)
            # not sure if this is right, but need to combine all probabilities if action space >1
            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=1)
            entropy_term = - self.actor.alpha * log_prob

            y = re_n + self.gamma * (1-terminal_n) * (torch.minimum(targ_q_1, targ_q_2) + entropy_term)

        # abit inefficient since we r forward passing twice.
        # possible to just stack q_1 and q_2, and repeat y twice, to get MSE loss
        # cos grad of q_1 wont affect grad of q_2
        q_1, q_2 = self.critic(ob_no, ac_na)
        self.critic.optimizer.zero_grad()
        critic_loss = self.critic.loss(y, q_1)
        critic_loss.backward()
        self.critic.optimizer.step()

        q_1, q_2 = self.critic(ob_no, ac_na)
        self.critic.optimizer.zero_grad()
        critic_loss = self.critic.loss(y, q_2)
        critic_loss.backward()
        self.critic.optimizer.step()
        ### my code ends here ###

        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor

        # 4. gather losses for logging

        ### my code starts here ###
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)

        loss = OrderedDict()
        for i in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        loss['Critic_Loss'] = critic_loss.item()

        if not self.training_step % self.critic_target_update_frequency:
            sac_utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
            sac_utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)

        if not self.training_step % self.actor_update_frequency:
            for j in range(self.agent_params['num_actor_updates_per_agent_update']):
                actor_loss, alpha_loss, temp = self.actor.update(ob_no, self.critic)
            loss['Actor_Loss'] = actor_loss.item()
            loss['Alpha_Loss'] = alpha_loss.item()
            loss['Temperature'] = temp


        self.training_step += 1
        ### my code ends here ###

        # loss = OrderedDict()
        # loss['Critic_Loss'] = TODO
        # loss['Actor_Loss'] = TODO
        # loss['Alpha_Loss'] = TODO
        # loss['Temperature'] = TODO

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
