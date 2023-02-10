from network import FeedForwardNN
from torch.distributions import MultivariateNormal
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gym

class PPO:
    def __init__(self, env) -> None:
        self.__init_hyperparameters()

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), self.lr)
        self.critic_optim = Adam(self.critic.parameters(), self.lr)

    def learn(self, total_timesteps):
        cnt_timestep = 0
        while cnt_timestep < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            cnt_timestep += np.sum(batch_lens)
            v_phi, _ = self.eval(batch_obs, batch_acts)
            a_k = batch_rtgs - v_phi.detach()
            a_k = (a_k - a_k.mean()) / (a_k.std() + 1e-10) ##小技巧，作者说把优势值归一化，可以提高稳定性，加1e-10只是为了防止被除数为0

            for _ in range(self.n_updates_per_itration):
                v_phi, curr_log_probs = self.eval(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * a_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * a_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(v_phi, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            

    def __init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_itration = 5
        self.clip = 0.2
        self.lr = 0.005

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_lens = []

        t = 0
        render = False
        while t < self.timesteps_per_batch:
            ep_rewards = []
            obs = self.env.reset()
            done = False
            if (t / self.max_timesteps_per_episode) % 10000== 0:
                render = True
            else:
                render = False
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                if render:
                    env.render()
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, reward, done, _ = self.env.step(action)
                ep_rewards.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break
            batch_lens.append(ep_t + 1)
            batch_rewards.append(ep_rewards)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        return batch_obs, batch_acts, batch_log_probs, batch_rewards_to_go, batch_lens

    def get_action(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def compute_rewards_to_go(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discount_reward = 0
            for rew in reversed(ep_rews):
                discount_reward = rew + discount_reward * self.gamma
                batch_rtgs.insert(0, discount_reward) ##这里用insert是为了反向挂载累计回报
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def eval(self, batch_obs, batch_acts):
        v = self.critic(batch_obs).squeeze()##计算V函数
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return v, log_probs


env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(1000000)
env.close()