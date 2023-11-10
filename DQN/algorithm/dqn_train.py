import torch
import gym
import random
import numpy as np

from DQN.algorithm import HyperParams
from DQN.algorithm.ReplayBuffer import ReplayBuffer
from DQN.algorithm.dqn import DQN
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(HyperParams.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                env.render()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (HyperParams.num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    env.close()
    agent.save()
    return return_list


if __name__ == '__main__':
    env = gym.make(HyperParams.env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(HyperParams.buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, HyperParams.hidden_dim, action_dim, HyperParams.lr, HyperParams.gamma, HyperParams.epsilon,
                HyperParams.target_update, HyperParams.device)
    return_list = train_DQN(agent, env, HyperParams.num_episodes, replay_buffer, HyperParams.minimal_size,
                            HyperParams.batch_size)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(HyperParams.env_name))
    plt.savefig('../results/native_return.png')

    mv_return = utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(HyperParams.env_name))
    plt.savefig('../results/moving_average_return.png')
