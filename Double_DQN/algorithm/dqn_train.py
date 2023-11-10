import gym
import numpy as np
from tqdm import tqdm
import random
import torch

from Double_DQN.algorithm import HyperParams
from Double_DQN.algorithm.ReplayBuffer import ReplayBuffer
from Double_DQN.algorithm.double_dqn import DQN
import matplotlib.pyplot as plt
import utils


def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (discrete_action / (action_dim - 1)) * (action_upbound - action_lowbound)


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                env.render()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = dis_to_con(action, env, agent.action_dim)
                    next_state, reward, done, _ = env.step([action_continuous])
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    env.close()
    agent.save('../model/dqn_model.pth')
    return return_list, max_q_value_list


if __name__ == '__main__':
    env = gym.make(HyperParams.env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(HyperParams.buffer_size)
    state_dim = env.observation_space.shape[0]
    # 将连续动作分成11个离散动作
    action_dim = 11
    agent = DQN(state_dim, HyperParams.hidden_dim, action_dim, HyperParams.lr, HyperParams.gamma, HyperParams.epsilon,
                HyperParams.target_update, HyperParams.device, 'VanillaDQN')
    return_list, max_q_value_list = train_DQN(agent, env, HyperParams.num_episodes, replay_buffer,
                                              HyperParams.minimal_size, HyperParams.batch_size)
    episodes_list = list(range(len(return_list)))
    mv_return = utils.moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(HyperParams.env_name))
    plt.savefig('../results/DQN_moving_average_return.png')
    # 清除当前figure上的所有内容
    plt.clf()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format(HyperParams.env_name))
    plt.savefig('../results/DQN_max_q_value_list.png')
