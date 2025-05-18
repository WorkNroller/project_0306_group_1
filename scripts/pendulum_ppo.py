import gymnasium as gym
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
from torch.distributions import Normal
from tqdm import tqdm
import os

# Парсинг аргументов
parser = argparse.ArgumentParser(description='PPO for Pendulum')
parser.add_argument('--num-iterations', type=int, default=1000, help='number of iterations for learning')
parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs for updating policy')
parser.add_argument('--clip-ratio', type=float, default=0.2, help='clip value for PPO loss')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--value-coef', type=float, default=0.5, help='value loss coefficient')
parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy loss coefficient')
parser.add_argument('--sub-batch-size', type=int, default=32, help='size of sub-samples')
parser.add_argument('--steps', type=int, default=2048, help='number of steps per trajectory')
parser.add_argument('--gae-lambda', type=float, default=0.95, help='lambda for general advantage estimation')
parser.add_argument('--normalize-advantages', action='store_true', default=True, help='normalize advantages')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, help='interval between training status logs')
args = parser.parse_args()

# Параметры
lr = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализация окружения
env = gym.make('Pendulum-v1', render_mode='rgb_array')

env.reset(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Размеры пространства состояний и действий
state_dim = env.observation_space.shape[0]  # 3
action_dim = env.action_space.shape[0]  # 1
action_low = float(env.action_space.low[0])  # -2.0
action_high = float(env.action_space.high[0])  # 2.0

# Класс буфера
class RolloutBuffer:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_timesteps = []
        self.episode_reward = 0
        self.episode_length = 0
        self.current_steps = 0
        self.global_timesteps = 0

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.episode_reward = 0
        self.episode_length = 0
        self.current_steps = 0

    def add(self, state, action, reward, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.episode_reward += reward
        self.episode_length += 1
        self.current_steps += 1
        self.global_timesteps += 1

    def end_episode(self):
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self.episode_length)
        self.episode_timesteps.append(self.global_timesteps)
        self.episode_reward = 0
        self.episode_length = 0

    def is_full(self):
        return self.current_steps >= self.max_steps

    def get_data(self):
        return {
            "states": torch.FloatTensor(self.states).to(device),
            "actions": torch.FloatTensor(self.actions).to(device),
            "rewards": self.rewards,
            "log_probs": torch.FloatTensor(self.log_probs).to(device),
            "dones": self.dones,
        }

# Модель актора
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.mean_net(state) * action_high
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_dist(self, state):
        mean, std = self.forward(state)
        return Normal(mean, std)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = self.get_dist(state)
        action = dist.sample()
        action = torch.clamp(action, min=action_low, max=action_high)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy()[0], log_prob.item()

# Модель критика
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.network(state)

# Вычисление возвратов и преимуществ с GAE
def compute_returns_and_advantages(rewards, dones, values):
    returns = []
    advantages = []
    last_gae_lam = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - dones[t]
            next_value = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_value = values[t + 1]

        delta = rewards[t] + args.gamma * next_value * next_nonterminal - values[t]
        last_gae_lam = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae_lam
        advantages.insert(0, last_gae_lam)
        returns.insert(0, last_gae_lam + values[t])
        next_value = values[t]

    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    if args.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages

# Обновление политики
def update_ppo(states, actions, log_probs_old, returns, advantages):
    for _ in range(args.num_epochs):
        indices = np.random.permutation(len(states))
        for start in range(0, len(states), args.sub_batch_size):
            batch_indices = indices[start:start + args.sub_batch_size]
            state_batch = states[batch_indices]
            action_batch = actions[batch_indices]
            log_prob_old_batch = log_probs_old[batch_indices]
            return_batch = returns[batch_indices]
            advantage_batch = advantages[batch_indices]

            dist = actor.get_dist(state_batch)
            log_prob = dist.log_prob(action_batch).sum(dim=-1)
            value = critic(state_batch).squeeze()

            ratio = torch.exp(log_prob - log_prob_old_batch)
            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * advantage_batch
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = dist.entropy().mean()
            actor_loss = actor_loss - args.entropy_coef * entropy_loss

            critic_loss = args.value_coef * nn.MSELoss()(value, return_batch)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

# Сбор траекторий
def collect_trajectories(actor, buffer):
    state, _ = env.reset()
    buffer.reset()
    while not buffer.is_full():
        action, log_prob = actor.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(state, action, reward, log_prob, done)
        state = next_state
        if done:
            buffer.end_episode()
            state, _ = env.reset()

    return buffer.get_data()

# Построение графика timesteps vs. episodic return
def plot_timesteps_vs_rewards(timesteps, rewards, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, rewards, label='Episodic Return', alpha=0.5)
    if len(rewards) >= 100:
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(timesteps, means.numpy(), label='100-episode Moving Average', linestyle='--')
    plt.xlabel('Timesteps')
    plt.ylabel('Episodic Return')
    plt.title('Timesteps vs. Episodic Return')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Основной цикл обучения
def train_ppo():
    global actor, critic, actor_optimizer, critic_optimizer
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim).to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    buffer = RolloutBuffer(args.steps)
    for iteration in tqdm(range(args.num_iterations), desc="Training"):
        batch = collect_trajectories(actor, buffer)
        states = batch["states"]
        actions = batch["actions"]
        log_probs_old = batch["log_probs"]
        with torch.no_grad():
            values = critic(states).squeeze()
        returns, advantages = compute_returns_and_advantages(batch["rewards"], batch["dones"], values)
        update_ppo(states, actions, log_probs_old, returns, advantages)

        if (iteration + 1) % args.log_interval == 0:
            avg_reward = np.mean(buffer.episode_rewards[-args.log_interval:]) if buffer.episode_rewards else 0
            avg_length = np.mean(buffer.episode_lengths[-args.log_interval:]) if buffer.episode_lengths else 0
            print(f"Iteration {iteration + 1}:")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Episode Length: {avg_length:.2f}")

    return buffer.episode_timesteps, buffer.episode_rewards

if __name__ == '__main__':
    os.makedirs("results", exist_ok=True)
    timesteps, rewards = train_ppo()
    plot_timesteps_vs_rewards(timesteps, rewards, "results/timesteps_vs_rewards.png")
    env.close()