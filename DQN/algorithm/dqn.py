import numpy as np
import torch
import torch.nn.functional as F

from DQN.algorithm.Qnet import Qnet


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        # 输出维度
        self.action_dim = action_dim
        # 训练设备：cpu/gpu
        self.device = device
        # q network
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # target network
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
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

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # Q值
        q_values = self.q_net(states).gather(1, actions)
        # 下一个状态的最大Q值
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
    def save(self):
        torch.save(self.q_net.state_dict(), '../model/dqn_model.pth')

    # 读取模型
    def load(self):
        self.q_net.load_state_dict(torch.load('../model/dqn_model.pth'))
