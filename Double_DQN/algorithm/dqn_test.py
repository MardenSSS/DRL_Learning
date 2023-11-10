import gym
from Double_DQN.algorithm import HyperParams
from Double_DQN.algorithm.double_dqn import DQN
from Double_DQN.algorithm.double_dqn_train import dis_to_con

if __name__ == '__main__':
    env = gym.make(HyperParams.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 11
    state = env.reset()
    total_reward = 0
    agent = DQN(state_dim, HyperParams.hidden_dim, action_dim, HyperParams.lr, HyperParams.gamma, HyperParams.epsilon,
                HyperParams.target_update, HyperParams.device, 'VanillaDQN')
    agent.load('../model/dqn_model.pth')
    while True:
        env.render()
        action = agent.take_action(state)
        action_continuous = dis_to_con(action, env, agent.action_dim)
        next_state, reward, done, _ = env.step([action_continuous])
        total_reward += reward
        state = next_state
        if done:
            break
    env.close()
    print('Total reward:', total_reward)
