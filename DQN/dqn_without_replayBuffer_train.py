import torch
import gym
import random
import numpy as np
from dqn_without_replayBuffer import DQN
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils

if __name__ == '__main__':
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    return_list = []

    for i in range(50):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                env.render()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    episode_return += reward

                    # print('state', state)
                    # print('action', action)
                    # print('reward', reward)
                    # print('next_state', next_state)
                    # print('done', done)
                    reshape_state = state.reshape(1, -1)
                    reshape_action = (action,)
                    reshape_reward = (reward,)
                    reshape_next_state = next_state.reshape(1, -1)
                    reshape_done = (done,)
                    # print('reshape_state', state)
                    # print('reshape_action', action)
                    # print('reshape_reward', reward)
                    # print('reshape_next_state', next_state)
                    # print('reshape_done', done)
                    agent.update(reshape_state, reshape_action, reshape_reward, reshape_next_state, reshape_done)

                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    env.close()
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.savefig('./results/return_without_replayBuffer.png')

    mv_return = utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.savefig('./results/moving_average_return_without_replayBuffer.png')

    agent.save()
