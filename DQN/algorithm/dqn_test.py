import gym
from DQN.algorithm import HyperParams

from DQN.algorithm.dqn import DQN

if __name__ == '__main__':
    env = gym.make(HyperParams.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    state = env.reset()
    total_reward = 0
    agent = DQN(state_dim, HyperParams.hidden_dim, action_dim, HyperParams.lr, HyperParams.gamma, HyperParams.epsilon,
                HyperParams.target_update, HyperParams.device)
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
