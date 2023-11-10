import numpy as np
import torch
import torch.nn.functional as F

from Dueling_DQN.algorithm.VAnet import VAnet, Qnet


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                 dqn_type='VanillaDQN'):
        # 输出维度
        self.action_dim = action_dim
        # 训练设备：cpu/gpu
        self.device = device
        if dqn_type == 'DuelingDQN':  # Dueling DQN采取不一样的网络框架
            self.q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = VAnet(state_dim, hidden_dim, self.action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
            self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        # 折扣因子
        self.gamma = gamma
        # epsilon-贪婪策略
        self.epsilon = epsilon
        # 目标网络更新频率
        self.target_update = target_update
        # 计数器，记录更新次数
        self.count = 0
        # DQN算法类型，默认原版DQN
        self.dqn_type = dqn_type

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # Q值
        q_values = self.q_net(states).gather(1, actions)
        # 下一个状态的最大Q值
        if self.dqn_type == 'DoubleDQN':  # Double DQN
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:  # DQN
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # TD target
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # 均方误差损失函数
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()
        # 反向传播更新参数
        dqn_loss.backward()
        self.optimizer.step()
        if self.count % self.target_update == 0:
            # 更新目标网络
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

    # 保存模型
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    # 读取模型
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
