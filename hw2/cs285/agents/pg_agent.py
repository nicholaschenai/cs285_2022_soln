import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data 
        # using helper functions to compute qvals and advantages, and
        # return the train_log obtained from updating the policy

        ### my code starts here ###
        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, np.concatenate(q_values), terminals)
        train_log = self.actor.update(observations, actions, advantages, q_values=q_values)
        ### my code ends here ###

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # Note: rewards_list is a list of lists of rewards with the inner list
        # being the list of rewards for a single trajectory.
        
        # HINT: use the helper functions self._discounted_return and
        # self._discounted_cumsum (you will need to implement these).

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory

        # Note: q_values should first be a 2D list where the first dimension corresponds to 
        # trajectories and the second corresponds to timesteps, 
        # then flattened to a 1D numpy array.

        # MEGA NOTE TO SELF: the original hw is slightly wrong--
        # at the update portion, we only have transitions in 1D, but we need the num
        # trajectories so we can average over. as such, ill leave q_val to be a list of arrays, ill
        # concat later so that i can extract num trajectories from q_val
        if not self.reward_to_go:
            # TODO
            ### my code starts here ###
            # q_values = np.concatenate([self._discounted_return(rewards) for rewards in rewards_list])
            q_values = [self._discounted_return(rewards) for rewards in rewards_list]
            ### my code ends here ###

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            # TODO
            ### my code starts here ###
            # q_values = np.concatenate([self._discounted_cumsum(rewards) for rewards in rewards_list])
            q_values = [self._discounted_cumsum(rewards) for rewards in rewards_list]
            ### my code ends here ###

        return q_values

    def estimate_advantage(self, obs: np.ndarray, rews_list: np.ndarray, q_values: np.ndarray, terminals: np.ndarray):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            ## TODO: values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values
            # values = TODO

            ### my code starts here ###
            values_norm = (values_unnormalized - np.mean(values_unnormalized)) / np.std(values_unnormalized)
            q_mean = np.mean(q_values)
            q_std = np.std(q_values)
            values = values_norm * q_std + q_mean
            ### my code ends here ###


            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ## TODO: recursively compute advantage estimates starting from
                        ## timestep T.
                    ## HINT: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.

                    ### my code starts here ###
                    if terminals[i]:
                        advantages[i] = rews[i] - values[i]
                    else:
                        delta = rews[i] + self.gamma*values[i+1] - values[i]
                        advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i+1]
                    ### my code ends here ###

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                ## TODO: compute advantage estimates using q_values, and values as baselines
                # advantages = TODO
                ### my code starts here ###
                advantages = q_values - values
                ### my code ends here ###

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages to have a mean of zero
        # and a standard deviation of one
        if self.standardize_advantages:
            # advantages = TODO
            ### my code starts here ###
            # note: 2023 suggests norm within the batch so this is right
            # note: see asg 3 for cleaner implementation in ac_agent.py/estimate_advantage esp adding stability term
            mu = np.mean(advantages)
            sigma = np.std(advantages)
            advantages = (advantages-mu)/sigma
            ### my code ends here ###

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

            Note from 2023: Note that all entries of the output list should be the exact same because each sum
            is from 0 to T (and doesn't involve t)!
        """
        # TODO: implement
        ### my code starts here ###
        discounted_r = 0
        gamma_fac = 1
        for r_t_prime in rewards:
            discounted_r += r_t_prime * gamma_fac
            gamma_fac *= self.gamma
        list_of_discounted_returns = np.array([discounted_r for i in range(len(rewards))])
        # print(f'list_of_discounted_returns shape {list_of_discounted_returns.shape}')
        ### my code ends here ###

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # NOTE: do they mean index t?
        """
        # TODO: implement
        ### my code starts here ###
        # note that recursion trick! work backwards cos term at time t is reward + gamma * term at time t+1
        list_of_discounted_cumsums = np.array([r for r in rewards])
        for i in range(len(rewards)-2,-1,-1):
            list_of_discounted_cumsums[i] += self.gamma * list_of_discounted_cumsums[i+1]
        # print(f'list_of_discounted_cumsums shape {list_of_discounted_cumsums.shape}')
        ### my code ends here ###

        return list_of_discounted_cumsums
