from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term

        ### my code starts here ###
        # im not sure what this part wants as alpha is a coeff to entropy. ill just return the alpha value for this
        entropy = torch.exp(self.log_alpha)
        ### my code ends here ###

        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution

        ### my code starts here ###
        with torch.no_grad():
            observation = ptu.from_numpy(obs)[None]
            action_distribution: torch.distributions.Distribution = self(observation)
            if sample:
                action: torch.Tensor = action_distribution.sample()
            else:
                action: torch.Tensor = action_distribution.mean

            # action = ptu.to_numpy(action).squeeze(0)
            # dont squeeze cos eval_trajectory in utils.py alr extracts the dummy dimension
            action = ptu.to_numpy(action)
        ### my code ends here ###

        return action

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file

        # get original distribution from MLPPolicy forward
        batch_mean = self.mean_net(observation)
        # TODO: note to self: SAC uses log std dev as outputs of nn. cant seem to find that nn so i assume its a param
        clipped_log_std = torch.clamp(self.logstd, min=self.log_std_bounds[0], max=self.log_std_bounds[1])
        scale = torch.exp(clipped_log_std)
        action_distribution = sac_utils.SquashedNormal(batch_mean, scale)

        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        ### my code starts here ###
        self.optimizer.zero_grad()
        m = self(obs)
        r_actions = m.rsample() # rsample to use  reparam trick
        q_1, q_2 = critic(obs, r_actions)
        q_val = torch.minimum(q_1, q_2)

        r_log_prob = m.log_prob(r_actions)

        if len(r_log_prob.shape) > 1:
            r_log_prob = r_log_prob.sum(dim=1)

        alpha_log_prob = - self.alpha * r_log_prob

        # note: gradient ASCENT on actor
        actor_loss = -(q_val + alpha_log_prob).mean()
        actor_loss.backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        # seems like alpha loss doesnt use reparam trick. just normal actions
        # note: not sure if i should be sampling actions or just using actions from
        # replay buffer since the eqn is pi_t which implies policy at time t rather 
        # than the most updated policy? still, this method doesnt allow access to
        # ac_na so i assume we need to sample
        with torch.no_grad():
            m = self(obs)
            actions = m.sample()
            log_prob = m.log_prob(actions)
            if len(log_prob.shape) > 1:
                log_prob = log_prob.sum(dim=1)
        alpha_loss = (- self.alpha * log_prob - self.alpha * self.target_entropy).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        ### my code ends here ###

        return actor_loss, alpha_loss, self.alpha