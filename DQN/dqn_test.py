import gym
import torch

from DQN.dqn import DQN

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    lr = 2e-3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    state = env.reset()
    total_reward = 0
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    agent.load()
    while True:
        env.render()
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    env.close()
    print('Total reward:', total_reward)
