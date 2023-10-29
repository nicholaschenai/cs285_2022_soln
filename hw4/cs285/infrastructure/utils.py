import numpy as np
import time
import copy

############################################
############################################

def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0],0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac,0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states

def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def mean_squared_error(a, b):
    return np.mean((a-b)**2)

############################################
############################################

# def sample_trajectory(env, policy, max_path_length, render=False):
# TODO: get this from previous HW

# from cs285 2023
def sample_trajectory(
        env, policy, max_path_length: int, render: bool = False
):
    """Sample a rollout in the environment from a policy."""
    ob = env.reset()
    obs, acs, rewards, next_obs, dones, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        # render an image
        if render:
            if hasattr(env, "sim"):
                img = env.sim.render(camera_name="track", height=500, width=500)[::-1]
            else:
                img = env.render(mode="rgb_array")

            if isinstance(img, list):
                img = img[0]

            image_obs.append(
                cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC)
            )

        ac = policy.get_action(ob)

        # print(f'ac dim b4 {ac.shape}')
        if len(ac.shape)>1:
            ac = np.squeeze(ac, 0)
        # print(f'ac dim aft {ac.shape}')

        next_ob, rew, done, info = env.step(ac)

        steps += 1
        # only record a "done" into the replay buffer if not truncated
        done_not_truncated = (
                done and steps <= max_path_length and not info.get("TimeLimit.truncated", False)
        )

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        dones.append(done_not_truncated)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if done or steps > max_path_length:
            break

    episode_statistics = {"l": steps, "r": np.sum(rewards)}
    if "episode" in info:
        episode_statistics.update(info["episode"])

    env.close()

    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(dones, dtype=np.float32),
        "episode_statistics": episode_statistics,
    }


# def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
#     """
#         Collect rollouts using policy
#         until we have collected min_timesteps_per_batch steps
#     """
#     # TODO: get this from previous HW
#
#     return paths, timesteps_this_batch

# from cs285 2023
def sample_trajectories(
    env,
    policy,
    min_timesteps_per_batch: int,
    max_length: int,
    render: bool = False,
):
    """Collect rollouts using policy until we have collected min_timesteps_per_batch steps."""
    timesteps_this_batch = 0
    trajs = []
    while timesteps_this_batch < min_timesteps_per_batch:
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)

        # count steps
        timesteps_this_batch += len(traj["reward"])
    return trajs, timesteps_this_batch

# def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
#     """
#         Collect ntraj rollouts using policy
#     """
#     # TODO: get this from Piazza
#
#     return paths

# from cs285 2023
def sample_n_trajectories(
    env, policy, ntraj: int, max_length: int, render: bool = False
):
    """Collect ntraj rollouts."""
    trajs = []
    for i in range(ntraj):
        # collect rollout
        traj = sample_trajectory(env, policy, max_length, render)
        trajs.append(traj)
    return trajs

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def unnormalize(data, mean, std):
    return data*std+mean

def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data
